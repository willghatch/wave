// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Loop Invariant Sinking Pass
//
// Selectively sinks cheap-to-recompute loop-invariant operations into loop
// bodies to reduce register pressure.  Intended to run after LICM when
// hoisting has pushed VGPR pressure above the hardware limit.
//
// Uses CLONE semantics: candidates are cloned into the loop body with
// in-loop uses rewired to the clones.  Originals are erased if dead.
//
// Safety: only sinks an op when doing so does not create new cross-loop
// VGPR live ranges.  An op's VGPR operand is "safe" if it is:
//   (a) already used inside the loop, OR
//   (b) defined by another op that will also be sunk (transitive closure)
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-loop-sinking"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMLOOPSINKING
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

static bool isSinkableOp(Operation *op) {
  if (!isa<V_LSHRREV_B32, V_AND_B32, V_LSHLREV_B32, V_SUB_U32,
           V_BFE_U32, V_MUL_LO_U32, V_MIN_I32, V_ADD_U32,
           V_OR_B32, V_XOR_B32, V_LSHL_ADD_U32, V_CNDMASK_B32,
           V_LSHL_OR_B32>(op))
    return false;
  if (op->getNumResults() != 1)
    return false;
  return isVGPRType(op->getResult(0).getType());
}

static bool hasUseInsideRegion(Value val, Region *region) {
  for (OpOperand &use : val.getUses()) {
    if (region->isAncestor(use.getOwner()->getParentRegion()))
      return true;
  }
  return false;
}

static bool operandSafe(Value operand, Region *loopRegion,
                        const llvm::DenseSet<Operation *> &sinkSet) {
  if (!isVGPRType(operand.getType()))
    return true;
  if (hasUseInsideRegion(operand, loopRegion))
    return true;
  if (auto *defOp = operand.getDefiningOp()) {
    if (sinkSet.contains(defOp))
      return true;
  }
  return false;
}

static unsigned sinkIntoLoop(LoopOp loopOp) {
  Block &body = loopOp.getBodyBlock();
  Region *loopRegion = &loopOp.getBodyRegion();
  unsigned numSunk = 0;

  // Phase 1: collect sinkable ops defined outside the loop whose results
  // are used inside it.
  llvm::DenseSet<Operation *> candidates;
  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (loopRegion->isAncestor(defOp->getParentRegion()))
        continue;
      if (!isSinkableOp(defOp))
        continue;
      candidates.insert(defOp);
    }
  });

  if (candidates.empty())
    return 0;

  // Phase 2: expand to include transitive dependencies so that entire
  // computation chains (e.g. v_add -> v_lshrrev) can be sunk together.
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *op : llvm::SmallVector<Operation *>(candidates.begin(),
                                                        candidates.end())) {
      for (Value operand : op->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp)
          continue;
        if (loopRegion->isAncestor(defOp->getParentRegion()))
          continue;
        if (!isSinkableOp(defOp))
          continue;
        if (candidates.insert(defOp).second)
          changed = true;
      }
    }
  }

  // Phase 3: iteratively remove candidates whose VGPR operands are
  // neither already live inside the loop nor defined by another candidate.
  changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> toRemove;
    for (Operation *op : candidates) {
      for (Value operand : op->getOperands()) {
        if (!operandSafe(operand, loopRegion, candidates)) {
          toRemove.push_back(op);
          break;
        }
      }
    }
    for (Operation *op : toRemove) {
      if (candidates.erase(op))
        changed = true;
    }
  }

  if (candidates.empty())
    return 0;

  // Phase 4: collect in topological (program) order by walking ancestor
  // blocks from outermost to innermost.
  SmallVector<Block *> ancestorBlocks;
  {
    Operation *cur = loopOp.getOperation();
    while (cur) {
      if (Block *b = cur->getBlock())
        ancestorBlocks.push_back(b);
      cur = cur->getParentOp();
    }
  }
  std::reverse(ancestorBlocks.begin(), ancestorBlocks.end());

  SmallVector<Operation *> toSink;
  for (Block *block : ancestorBlocks) {
    for (Operation &op : *block) {
      if (candidates.count(&op))
        toSink.push_back(&op);
    }
  }

  // Phase 5: clone into loop body and rewire in-loop uses.
  IRMapping mapping;
  OpBuilder builder = OpBuilder::atBlockBegin(&body);

  for (Operation *op : toSink) {
    Value origResult = op->getResult(0);

    Operation *clone = builder.clone(*op, mapping);
    Value clonedResult = clone->getResult(0);

    SmallVector<OpOperand *> usesToReplace;
    for (OpOperand &use : origResult.getUses()) {
      if (loopRegion->isAncestor(use.getOwner()->getParentRegion()))
        usesToReplace.push_back(&use);
    }
    for (OpOperand *use : usesToReplace)
      use->set(clonedResult);

    mapping.map(origResult, clonedResult);
    ++numSunk;
  }

  // Phase 6: erase dead originals.
  for (auto it = toSink.rbegin(); it != toSink.rend(); ++it) {
    Operation *op = *it;
    if (op->getResult(0).use_empty())
      op->erase();
  }

  LLVM_DEBUG(llvm::dbgs() << "[SINK] cloned " << numSunk << " ops into loop\n");
  return numSunk;
}

struct LoopSinkingPass
    : public waveasm::impl::WAVEASMLoopSinkingBase<LoopSinkingPass> {
  using WAVEASMLoopSinkingBase::WAVEASMLoopSinkingBase;

  void runOnOperation() override {
    SmallVector<LoopOp> loops;
    getOperation()->walk<WalkOrder::PostOrder>(
        [&](LoopOp loopOp) { loops.push_back(loopOp); });
    for (auto loopOp : loops)
      numOpsSunk += sinkIntoLoop(loopOp);
  }
};

} // namespace
