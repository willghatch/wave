// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Shared Helper Functions for Operation Handlers
//===----------------------------------------------------------------------===//
//
// This file implements utility functions declared in Handlers.h that are
// shared across multiple handler files (ArithHandlers, AffineHandlers,
// MemRefHandlers, AMDGPUHandlers, etc.).
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace waveasm {

//===----------------------------------------------------------------------===//
// isLDSMemRef
//===----------------------------------------------------------------------===//

bool isLDSMemRef(MemRefType memrefType) {
  auto memSpace = memrefType.getMemorySpace();
  if (!memSpace)
    return false;

  // Check for gpu.address_space<workgroup> attribute
  if (auto gpuSpace = dyn_cast<gpu::AddressSpaceAttr>(memSpace)) {
    return gpuSpace.getValue() == gpu::AddressSpace::Workgroup;
  }
  // Also check for integer address space (3 on AMDGPU)
  if (auto intAttr = dyn_cast<IntegerAttr>(memSpace)) {
    return intAttr.getInt() == 3;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// getElementBytes
//===----------------------------------------------------------------------===//

int64_t getElementBytes(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth() / 8;
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8;
  return 4;
}

//===----------------------------------------------------------------------===//
// computeBufferSizeFromMemRef
//===----------------------------------------------------------------------===//

int64_t computeBufferSizeFromMemRef(MemRefType memrefType) {
  static constexpr int64_t kOOBSentinel = 0x7FFFFFFE;

  // For ranked memrefs with fully static shapes, compute the real buffer size
  // in bytes.  This enables hardware bounds checking for out-of-bounds loads
  // (e.g. during drain iterations of pipelined loops with
  // eliminate_epilogue=True).  Reads past NUM_RECORDS return zero on GFX9+.
  if (memrefType.hasStaticShape() && memrefType.getRank() > 0) {
    int64_t numElements = 1;
    for (auto dim : memrefType.getShape()) {
      if (dim <= 0)
        return kOOBSentinel;
      if (dim > kOOBSentinel / std::max(numElements, int64_t(1)))
        return kOOBSentinel;
      numElements *= dim;
    }
    int64_t elementBits = memrefType.getElementTypeBitWidth();
    int64_t totalBytes = (numElements * elementBits + 7) / 8;
    if (totalBytes <= 0 || totalBytes > kOOBSentinel)
      return kOOBSentinel;
    return totalBytes;
  }

  // Rank-0 or dynamic shapes: fall back to the OOB sentinel so that the
  // Python frontend's OOB-index scheme still works (sentinel index at
  // 0x7FFFFFFF is one byte past the SRD range).
  return kOOBSentinel;
}

//===----------------------------------------------------------------------===//
// isPowerOf2 / log2
//===----------------------------------------------------------------------===//

bool isPowerOf2(int64_t val) { return val > 0 && (val & (val - 1)) == 0; }

int64_t log2(int64_t val) {
  int64_t result = 0;
  while ((1LL << result) < val)
    ++result;
  return result;
}

//===----------------------------------------------------------------------===//
// getArithConstantValue
//===----------------------------------------------------------------------===//

std::optional<int64_t> getArithConstantValue(Value val) {
  IntegerAttr attr;
  if (mlir::matchPattern(val, mlir::m_Constant(&attr)))
    return attr.getInt();
  return std::nullopt;
}

} // namespace waveasm
