// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Region Rematerialization Pass
//
// Clones cheap VALU ops into the regions where they are used, shortening
// their live ranges to reduce VGPR pressure.  Specifically targets the
// dynamic-shape pattern:
//
//   <cheap ops defined here>      // live range spans entire if + remainder
//   %results = scf.if %cond {
//       // pipelined loop
//   } else {
//       yield zeros
//   }
//   // remainder loop
//
// After rematerialization, copies of the cheap ops are placed at the top
// of each region that uses them, and the original uses are rewired to the
// clones.  If the original has no remaining uses, it becomes dead and will
// be eliminated by subsequent canonicalization.
//===----------------------------------------------------------------------===//

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMTypes.h"
#include "waveasm/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "waveasm-region-remat"

using namespace mlir;
using namespace waveasm;

namespace waveasm {
#define GEN_PASS_DEF_WAVEASMREGIONREMATERIALIZATION
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

// Collect uses of a value that are within a given region (at any depth).
static SmallVector<OpOperand *> getUsesInRegion(Value val, Region *region) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : val.getUses()) {
    if (region->isAncestor(use.getOwner()->getParentRegion()))
      uses.push_back(&use);
  }
  return uses;
}

// Check if all operands of an op are available inside the target region.
// An operand is available if it is defined outside the region (dominates it)
// or is already rematerialized inside it (tracked by the mapping).
static bool operandsAvailable(Operation *op, Region *targetRegion,
                              const IRMapping &mapping) {
  for (Value operand : op->getOperands()) {
    // If we have a mapping for this operand, it's available.
    if (mapping.contains(operand))
      continue;
    // If the operand is defined outside the region's parent, it dominates.
    if (auto defOp = operand.getDefiningOp()) {
      if (!targetRegion->isAncestor(defOp->getParentRegion()))
        continue;
    } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (!targetRegion->isAncestor(parentOp->getParentRegion()) &&
          parentOp != targetRegion->getParentOp())
        continue;
      // Block argument of the region's parent op (e.g., loop block arg)
      // is available inside the region.
      if (parentOp == targetRegion->getParentOp())
        continue;
    }
    // Constants/immediates are always available.
    if (auto defOp = operand.getDefiningOp()) {
      if (isa<ConstantOp>(defOp))
        continue;
    }
    return false;
  }
  return true;
}

// Rematerialize candidates into a target region.  Returns the number of
// ops cloned.  The mapping tracks original -> cloned values so that
// chains of candidates reuse cloned operands rather than the originals.
static unsigned rematerializeIntoRegion(
    ArrayRef<Operation *> candidates, Region *targetRegion, IRMapping &mapping) {
  if (candidates.empty() || targetRegion->empty())
    return 0;

  Block &targetBlock = targetRegion->front();
  OpBuilder builder = OpBuilder::atBlockBegin(&targetBlock);
  unsigned numCloned = 0;

  for (Operation *op : candidates) {
    Value origResult = op->getResult(0);

    // Only clone if there are uses in this region.
    auto uses = getUsesInRegion(origResult, targetRegion);
    if (uses.empty())
      continue;

    // Check that all operands are available.
    if (!operandsAvailable(op, targetRegion, mapping))
      continue;

    // Clone the op into the target region.
    Operation *clone = builder.clone(*op, mapping);
    Value clonedResult = clone->getResult(0);

    // Rewire uses inside the target region to use the clone.
    for (OpOperand *use : uses) {
      use->set(clonedResult);
    }

    // Track in mapping for dependent ops.
    mapping.map(origResult, clonedResult);
    ++numCloned;

    LLVM_DEBUG(llvm::dbgs() << "[REMAT] cloned into region: "
                            << op->getName() << "\n");
  }

  return numCloned;
}

// Collect all "sibling regions" that need rematerialization.
// For an IfOp, this includes the then-region and else-region.
// For a LoopOp following the IfOp, the loop body region.
struct RegionTarget {
  Region *region;
  Operation *parentOp;
};

struct RegionRematerializationPass
    : public waveasm::impl::WAVEASMRegionRematerializationBase<
          RegionRematerializationPass> {
  using WAVEASMRegionRematerializationBase::
      WAVEASMRegionRematerializationBase;

  void runOnOperation() override {
    Operation *module = getOperation();

    module->walk([&](ProgramOp program) {
      processProgram(program);
    });
  }

private:
  void processProgram(ProgramOp program) {
    // Find all IfOp instances.  For each, look for a following LoopOp
    // (the remainder loop).  Collect regions that could benefit from
    // rematerialization.
    SmallVector<std::pair<IfOp, SmallVector<RegionTarget>>> targets;

    program.walk([&](IfOp ifOp) {
      SmallVector<RegionTarget> regions;

      // Then-region
      if (!ifOp.getThenRegion().empty())
        regions.push_back({&ifOp.getThenRegion(), ifOp});

      // Else-region
      if (!ifOp.getElseRegion().empty())
        regions.push_back({&ifOp.getElseRegion(), ifOp});

      // Look for a LoopOp that follows the IfOp in the same block.
      if (Operation *next = ifOp->getNextNode()) {
        if (auto loopOp = dyn_cast<LoopOp>(next)) {
          regions.push_back({&loopOp.getBodyRegion(), loopOp});
        }
      }

      if (!regions.empty())
        targets.push_back({ifOp, std::move(regions)});
    });

    for (auto &[ifOp, regions] : targets) {
      // Collect cheap VALU ops defined before the IfOp in the same
      // block, in program order.
      SmallVector<Operation *> candidates;
      Block *parentBlock = ifOp->getBlock();
      for (Operation &op : *parentBlock) {
        if (&op == ifOp.getOperation())
          break;
        if (isCheapVALUOp(&op)) {
          Value result = op.getResult(0);
          // Only consider ops whose results are used inside at least
          // one of the target regions (and possibly outside too).
          bool usedInTarget = false;
          for (auto &rt : regions) {
            if (!getUsesInRegion(result, rt.region).empty()) {
              usedInTarget = true;
              break;
            }
          }
          if (usedInTarget)
            candidates.push_back(&op);
        }
      }

      if (candidates.empty())
        continue;

      LLVM_DEBUG(llvm::dbgs() << "[REMAT] found " << candidates.size()
                              << " candidates before IfOp\n");

      // Clone candidates into each target region.
      for (auto &rt : regions) {
        IRMapping mapping;
        unsigned n = rematerializeIntoRegion(candidates, rt.region, mapping);
        numOpsRematerialized += n;
        LLVM_DEBUG(llvm::dbgs() << "[REMAT] cloned " << n
                                << " ops into region of "
                                << rt.parentOp->getName() << "\n");
      }

      // After cloning into all regions, check if any original op has
      // no remaining uses.  If so, it's dead and can be erased.
      for (auto it = candidates.rbegin(); it != candidates.rend(); ++it) {
        Operation *op = *it;
        if (op->getResult(0).use_empty())
          op->erase();
      }
    }
  }
};

} // namespace
