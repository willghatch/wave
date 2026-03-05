// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Loop Invariant Sinking Pass
//
// Selectively sinks cheap-to-recompute loop-invariant operations back into
// loop bodies to reduce register pressure.  Intended to run after LICM when
// hoisting has pushed VGPR pressure above the hardware limit.
//
// Only sinks operations that:
//   - Have the ArithmeticOp trait (pure VALU/SALU instructions)
//   - Produce a single VGPR result (not SGPR, not multi-result)
//   - Have all operands available inside the loop (either loop-invariant
//     values that dominate the loop, or block arguments)
//   - Are used only inside the loop body (not used outside)
//
// Net effect: trades a small amount of extra VALU work per iteration for
// reduced register pressure, since the sunk value is live only for the
// span of its uses within the loop body rather than the entire loop.
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

// An op is a candidate for sinking if it has the ArithmeticOp trait,
// produces exactly one VGPR result, and is not an MFMA.
static bool isCheapVALUOp(Operation *op) {
  if (!op->hasTrait<mlir::OpTrait::ArithmeticOp>())
    return false;
  if (op->hasTrait<mlir::OpTrait::MFMAOp>())
    return false;
  if (op->getNumResults() != 1)
    return false;
  return isVGPRType(op->getResult(0).getType());
}

// True if all uses of val are inside the loop region (including nested
// regions like IfOp bodies).
static bool allUsesInsideLoop(Value val, LoopOp loopOp) {
  Region *loopRegion = &loopOp.getBodyRegion();
  for (OpOperand &use : val.getUses()) {
    Operation *user = use.getOwner();
    if (!loopRegion->isAncestor(user->getParentRegion()))
      return false;
  }
  return true;
}

// Find the first use of `val` in the loop body's top-level op list.
// Returns nullptr if no use is found directly in the body block.
static Operation *findFirstUseInBody(Value val, Block &body) {
  for (Operation &op : body) {
    // Check if this op directly uses val
    for (Value operand : op.getOperands()) {
      if (operand == val)
        return &op;
    }
    // Check nested regions (for IfOp, inner LoopOp, etc.) -- if val is
    // used inside a nested region, we want to sink to just before the
    // parent region-holding op in the body.
    for (Region &region : op.getRegions()) {
      for (Block &block : region) {
        for (Operation &nested : block) {
          SmallVector<Operation *, 4> worklist;
          worklist.push_back(&nested);
          while (!worklist.empty()) {
            Operation *cur = worklist.pop_back_val();
            for (Value operand : cur->getOperands()) {
              if (operand == val)
                return &op;
            }
            for (Region &r : cur->getRegions())
              for (Block &b : r)
                for (Operation &child : b)
                  worklist.push_back(&child);
          }
        }
      }
    }
  }
  return nullptr;
}

static unsigned sinkIntoLoop(LoopOp loopOp) {
  Block &body = loopOp.getBodyBlock();
  Region *loopRegion = &loopOp.getBodyRegion();
  unsigned numSunk = 0;

  // Collect ops to sink: ops defined immediately before the loop that are
  // cheap VALU, used only inside the loop, with all operands available.
  // We scan backwards from the loop to catch chains of sinkable ops.
  SmallVector<Operation *> toSink;

  // Gather all values used inside the loop that are defined outside it
  // by cheap VALU ops.
  llvm::DenseSet<Operation *> candidates;
  body.walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp)
        continue;
      if (!loopRegion->isAncestor(defOp->getParentRegion()) &&
          isCheapVALUOp(defOp) && allUsesInsideLoop(defOp->getResult(0), loopOp))
        candidates.insert(defOp);
    }
  });

  if (candidates.empty())
    return 0;

  // Topologically order the candidates so that if A's result feeds B,
  // A is sunk before B.
  // Walk the block containing the loop; candidates in the same block
  // are already in program order.
  Block *loopParentBlock = loopOp->getBlock();
  for (Operation &op : *loopParentBlock) {
    if (candidates.count(&op))
      toSink.push_back(&op);
  }

  for (Operation *op : toSink) {
    Value result = op->getResult(0);

    // Find insertion point: just before the first use in the body.
    Operation *firstUser = findFirstUseInBody(result, body);
    if (!firstUser)
      continue;

    // Move the op into the loop body, just before its first user.
    op->moveBefore(firstUser);
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
    for (auto loopOp : loops) {
      unsigned sunk = sinkIntoLoop(loopOp);
      numOpsSunk += sunk;
    }
  }
};

} // namespace
