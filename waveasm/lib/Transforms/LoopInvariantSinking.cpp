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
// For ops used ONLY inside the loop: moves them into the loop body.
// The value is recomputed each iteration, but its live range shrinks
// from [def, loop_terminator] to [new_position, use_within_iteration].
//
// For ops used both inside and outside the loop: clones them into the
// loop body.  In-loop uses are rewired to the clone, splitting the live
// range so the original need not survive across the loop body.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "waveasm-loop-sinking"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMLOOPSINKING
#include "waveasm/Transforms/Passes.h.inc"
} // namespace waveasm

namespace {

static bool isCheapVALUOp(Operation *op) {
  if (!op->hasTrait<mlir::OpTrait::ArithmeticOp>())
    return false;
  if (op->hasTrait<mlir::OpTrait::MFMAOp>())
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

static bool allUsesInsideRegion(Value val, Region *region) {
  for (OpOperand &use : val.getUses()) {
    if (!region->isAncestor(use.getOwner()->getParentRegion()))
      return false;
  }
  return true;
}

static Operation *findFirstUseInBody(Value val, Block &body) {
  for (Operation &op : body) {
    for (Value operand : op.getOperands())
      if (operand == val)
        return &op;
    for (Region &region : op.getRegions())
      if (hasUseInsideRegion(val, &region))
        return &op;
  }
  return nullptr;
}

static void replaceUsesInsideRegion(Value oldVal, Value newVal,
                                    Region *region) {
  SmallVector<OpOperand *> usesToReplace;
  for (OpOperand &use : oldVal.getUses()) {
    if (region->isAncestor(use.getOwner()->getParentRegion()))
      usesToReplace.push_back(&use);
  }
  for (OpOperand *use : usesToReplace)
    use->set(newVal);
}

// Recursively collect the transitive closure of cheap VALU operands
// defined outside the loop region.  The collected set includes all ops
// that would need to be cloned to avoid introducing new in-loop uses
// of original values (which would extend those values' live ranges).
static void collectTransitiveDeps(Operation *op, Region *loopRegion,
                                  llvm::DenseSet<Operation *> &deps) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;
    if (loopRegion->isAncestor(defOp->getParentRegion()))
      continue;
    if (!isCheapVALUOp(defOp))
      continue;
    if (deps.insert(defOp).second)
      collectTransitiveDeps(defOp, loopRegion, deps);
  }
}

static unsigned sinkIntoLoop(LoopOp loopOp) {
  Block &body = loopOp.getBodyBlock();
  Region *loopRegion = &loopOp.getBodyRegion();
  unsigned numSunk = 0;

  // Phase 1: find all cheap VALU ops defined outside the loop whose
  // results are used inside it.
  llvm::DenseSet<Operation *> directCandidates;
  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (loopRegion->isAncestor(defOp->getParentRegion()))
        continue;
      if (!isCheapVALUOp(defOp))
        continue;
      directCandidates.insert(defOp);
    }
  });

  if (directCandidates.empty())
    return 0;

  // Phase 2: expand to transitive closure of cheap VALU dependencies.
  // This ensures cloned ops can reference other clones rather than
  // pulling original outside-the-loop values into the loop.
  llvm::DenseSet<Operation *> allCandidates = directCandidates;
  for (Operation *op : directCandidates)
    collectTransitiveDeps(op, loopRegion, allCandidates);

  // Collect in topological (program) order.
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
      if (allCandidates.count(&op))
        toSink.push_back(&op);
    }
  }

  // Maps original values to their in-loop equivalents (whether moved
  // or cloned) so that chains of sunk ops use the right operands.
  llvm::DenseMap<Value, Value> inLoopValues;

  for (Operation *op : toSink) {
    Value origResult = op->getResult(0);

    bool usedInLoop = hasUseInsideRegion(origResult, loopRegion);
    bool neededByClone = false;
    for (OpOperand &use : origResult.getUses()) {
      Operation *user = use.getOwner();
      if (allCandidates.count(user) && !loopRegion->isAncestor(user->getParentRegion()))
        neededByClone = true;
    }
    if (!usedInLoop && !neededByClone)
      continue;

    // Find insertion point: just before the first use in the body,
    // or at the beginning of the body if only used by other clones.
    Operation *insertBefore = nullptr;
    if (usedInLoop)
      insertBefore = findFirstUseInBody(origResult, body);
    if (!insertBefore)
      insertBefore = &body.front();

    if (allUsesInsideRegion(origResult, loopRegion)) {
      op->moveBefore(insertBefore);
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        auto it = inLoopValues.find(op->getOperand(i));
        if (it != inLoopValues.end())
          op->setOperand(i, it->second);
      }
      inLoopValues[origResult] = origResult;
    } else {
      // Clone the op into the loop body.  In-loop uses are rewired to
      // the clone so the original no longer needs to survive across
      // the loop body, reducing its live range.  Because we clone the
      // full transitive dependency chain, the clone's operands
      // reference other clones, not the originals.
      OpBuilder builder(insertBefore);
      Operation *clone = builder.clone(*op);
      Value cloneResult = clone->getResult(0);
      for (unsigned i = 0; i < clone->getNumOperands(); ++i) {
        auto it = inLoopValues.find(clone->getOperand(i));
        if (it != inLoopValues.end())
          clone->setOperand(i, it->second);
      }
      replaceUsesInsideRegion(origResult, cloneResult, loopRegion);
      inLoopValues[origResult] = cloneResult;
    }
    ++numSunk;
  }

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
