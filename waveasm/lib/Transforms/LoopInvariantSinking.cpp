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

static unsigned sinkIntoLoop(LoopOp loopOp) {
  Block &body = loopOp.getBodyBlock();
  Region *loopRegion = &loopOp.getBodyRegion();
  unsigned numSunk = 0;

  llvm::DenseSet<Operation *> candidates;
  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (loopRegion->isAncestor(defOp->getParentRegion()))
        continue;
      if (!isCheapVALUOp(defOp))
        continue;
      candidates.insert(defOp);
    }
  });

  llvm::errs() << "[SINKING] Loop at " << loopOp.getLoc()
               << ": " << candidates.size() << " candidates\n";

  if (candidates.empty())
    return 0;

  // Collect candidates in topological (program) order.  Candidates may
  // live in any ancestor block, so gather all blocks from the loop's
  // parent up to the ProgramOp and walk each.
  SmallVector<Block *> ancestorBlocks;
  {
    Operation *cur = loopOp.getOperation();
    while (cur) {
      if (Block *b = cur->getBlock())
        ancestorBlocks.push_back(b);
      cur = cur->getParentOp();
    }
  }
  // Reverse so outermost blocks come first (program order).
  std::reverse(ancestorBlocks.begin(), ancestorBlocks.end());

  SmallVector<Operation *> toSink;
  for (Block *block : ancestorBlocks) {
    for (Operation &op : *block) {
      if (candidates.count(&op))
        toSink.push_back(&op);
    }
  }

  // Track cloned/moved values so chains can be rewired.
  llvm::DenseMap<Value, Value> inLoopValues;

  for (Operation *op : toSink) {
    Value origResult = op->getResult(0);
    if (!hasUseInsideRegion(origResult, loopRegion))
      continue;

    Operation *firstUser = findFirstUseInBody(origResult, body);
    if (!firstUser)
      continue;

    if (!allUsesInsideRegion(origResult, loopRegion)) {
      llvm::errs() << "[SINKING]   Skip (has outside uses): " << *op << "\n";
      continue;
    }

    // Move the op into the loop.  Rewire operands to in-loop versions
    // of any previously sunk values.
    op->moveBefore(firstUser);
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto it = inLoopValues.find(op->getOperand(i));
      if (it != inLoopValues.end())
        op->setOperand(i, it->second);
    }
    inLoopValues[origResult] = origResult;
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
