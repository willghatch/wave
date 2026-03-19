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
  // The Wave Python frontend computes an OOB sentinel element index as
  //   (valid_bytes + elem_bytes) / elem_bytes
  // where valid_bytes = (1 << 31) - 1 - elem_bytes.  The sentinel's byte
  // offset is sentinel_idx * elem_bytes which simplifies to
  //   ((valid_bytes + elem_bytes) / elem_bytes) * elem_bytes
  // For f32 (elem_bytes=4) the sentinel byte offset = 0x7FFFFFFC.
  // NUM_RECORDS must be <= sentinel_byte_offset so hardware treats the
  // sentinel as OOB (returning 0 for loads, dropping stores).
  // Using 0x7FFFFFFE for all types breaks f32: 0x7FFFFFFC < 0x7FFFFFFE
  // so the sentinel is "in bounds" and the hardware executes the access.
  int64_t elemBytes = getElementBytes(memrefType.getElementType());
  return (1LL << 31) - 1 - elemBytes;
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
