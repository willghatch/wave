// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Affine Dialect Handlers
//===----------------------------------------------------------------------===//
//
// This file implements handlers for Affine dialect operations:
//   - affine.apply
//
// The implementation includes:
//   - Thread ID upper bound simplification
//   - Bit range tracking for OR optimization
//   - Power-of-2 optimizations (shift instead of multiply/divide)
//   - v_lshl_or_b32 fusion for non-overlapping bit ranges
//
//===----------------------------------------------------------------------===//

#include "Handlers.h"

#include "waveasm/Dialect/WaveASMOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include <functional>

using namespace mlir;

namespace waveasm {

static bool isScalarValue(Value v) {
  Type ty = v.getType();
  return isa<SRegType>(ty) || isa<PSRegType>(ty) || isa<ImmType>(ty);
}

// Return the scalar (SGPR) version of a value if one exists.
// Only considers SGPR/PSReg types and scalarArgMap (VGPR->SGPR bindings).
// Does NOT convert ImmType to SGPR -- that would cause unwanted SALU paths
// for compile-time constants in the static case.
static std::optional<Value> tryGetScalar(Value v, TranslationContext &ctx) {
  if (isa<SRegType>(v.getType()) || isa<PSRegType>(v.getType()))
    return v;
  if (auto s = ctx.getScalarVersion(v))
    return *s;
  return std::nullopt;
}

// SALU Granlund-Montgomery unsigned division by constant.
static Value emitScalarConstantUnsignedDiv(OpBuilder &builder, Location loc,
                                           TranslationContext &ctx,
                                           Value numerator, int64_t divisor) {
  assert(divisor > 0 && "divisor must be positive");
  if (divisor == 1)
    return numerator;

  auto sregType = ctx.createSRegType();
  int S = 0;
  while ((1ULL << S) < static_cast<uint64_t>(divisor))
    S++;
  uint64_t M = (static_cast<__uint128_t>(1) << (32 + S)) / divisor + 1;

  if (M <= 0xFFFFFFFFULL) {
    auto mImm = ctx.createImmType(static_cast<int64_t>(M));
    Value mConst = ConstantOp::create(builder, loc, mImm, static_cast<int64_t>(M));
    Value mS = S_MOV_B32::create(builder, loc, sregType, mConst);
    Value hi = S_MUL_HI_U32::create(builder, loc, sregType, numerator, mS);
    auto sImm = ctx.createImmType(S);
    Value sConst = ConstantOp::create(builder, loc, sImm, S);
    return S_LSHR_B32::create(builder, loc, sregType, hi, sConst);
  }

  uint64_t Mlow = M - (1ULL << 32);
  auto mlowImm = ctx.createImmType(static_cast<int64_t>(Mlow));
  Value mlowConst = ConstantOp::create(builder, loc, mlowImm, static_cast<int64_t>(Mlow));
  Value mlowS = S_MOV_B32::create(builder, loc, sregType, mlowConst);
  Value t = S_MUL_HI_U32::create(builder, loc, sregType, numerator, mlowS);
  Value diff = S_SUB_U32::create(builder, loc, sregType, sregType, numerator, t).getDst();
  auto oneImm = ctx.createImmType(1);
  Value oneConst = ConstantOp::create(builder, loc, oneImm, 1);
  Value halfDiff = S_LSHR_B32::create(builder, loc, sregType, diff, oneConst);
  Value sum = S_ADD_U32::create(builder, loc, sregType, sregType, t, halfDiff).getDst();
  int64_t finalShift = S - 1;
  auto fsImm = ctx.createImmType(finalShift);
  Value fsConst = ConstantOp::create(builder, loc, fsImm, finalShift);
  return S_LSHR_B32::create(builder, loc, sregType, sum, fsConst);
}

static Value emitScalarConstantUnsignedMod(OpBuilder &builder, Location loc,
                                           TranslationContext &ctx,
                                           Value numerator, int64_t divisor) {
  auto sregType = ctx.createSRegType();
  Value quotient = emitScalarConstantUnsignedDiv(builder, loc, ctx, numerator, divisor);
  auto dImm = ctx.createImmType(divisor);
  Value dConst = ConstantOp::create(builder, loc, dImm, divisor);
  Value dS = S_MOV_B32::create(builder, loc, sregType, dConst);
  Value product = S_MUL_I32::create(builder, loc, sregType, quotient, dS);
  return S_SUB_U32::create(builder, loc, sregType, sregType, numerator, product).getDst();
}

static Value emitScalarConstantCeilDiv(OpBuilder &builder, Location loc,
                                       TranslationContext &ctx,
                                       Value numerator, int64_t divisor) {
  auto sregType = ctx.createSRegType();
  int64_t bias = divisor - 1;
  auto biasImm = ctx.createImmType(bias);
  Value biasConst = ConstantOp::create(builder, loc, biasImm, bias);
  Value biased = S_ADD_U32::create(builder, loc, sregType, sregType, biasConst, numerator).getDst();
  return emitScalarConstantUnsignedDiv(builder, loc, ctx, biased, divisor);
}

// Granlund-Montgomery unsigned division by constant (VALU).
static Value emitConstantUnsignedDiv(OpBuilder &builder, Location loc,
                                     Type vregType, TranslationContext &ctx,
                                     Value numerator, int64_t divisor) {
  assert(divisor > 0 && "divisor must be positive");
  if (divisor == 1)
    return numerator;

  int S = 0;
  while ((1ULL << S) < static_cast<uint64_t>(divisor))
    S++;
  uint64_t M = (static_cast<__uint128_t>(1) << (32 + S)) / divisor + 1;

  if (M <= 0xFFFFFFFFULL) {
    auto mImm = ctx.createImmType(static_cast<int64_t>(M));
    Value mConst = ConstantOp::create(builder, loc, mImm, static_cast<int64_t>(M));
    Value hi = V_MUL_HI_U32::create(builder, loc, vregType, numerator, mConst);
    auto sImm = ctx.createImmType(S);
    Value sConst = ConstantOp::create(builder, loc, sImm, S);
    return V_LSHRREV_B32::create(builder, loc, vregType, sConst, hi);
  }

  uint64_t Mlow = M - (1ULL << 32);
  auto mlowImm = ctx.createImmType(static_cast<int64_t>(Mlow));
  Value mlowConst = ConstantOp::create(builder, loc, mlowImm, static_cast<int64_t>(Mlow));
  Value t = V_MUL_HI_U32::create(builder, loc, vregType, numerator, mlowConst);
  Value diff = V_SUB_U32::create(builder, loc, vregType, numerator, t);
  auto oneImm = ctx.createImmType(1);
  Value oneConst = ConstantOp::create(builder, loc, oneImm, 1);
  Value halfDiff = V_LSHRREV_B32::create(builder, loc, vregType, oneConst, diff);
  Value sum = V_ADD_U32::create(builder, loc, vregType, t, halfDiff);
  int64_t finalShift = S - 1;
  auto fsImm = ctx.createImmType(finalShift);
  Value fsConst = ConstantOp::create(builder, loc, fsImm, finalShift);
  return V_LSHRREV_B32::create(builder, loc, vregType, fsConst, sum);
}

static Value emitConstantUnsignedMod(OpBuilder &builder, Location loc,
                                     Type vregType, TranslationContext &ctx,
                                     Value numerator, int64_t divisor) {
  Value quotient = emitConstantUnsignedDiv(builder, loc, vregType, ctx, numerator, divisor);
  auto dImm = ctx.createImmType(divisor);
  Value dConst = ConstantOp::create(builder, loc, dImm, divisor);
  Value product = V_MUL_LO_U32::create(builder, loc, vregType, quotient, dConst);
  return V_SUB_U32::create(builder, loc, vregType, numerator, product);
}

static Value emitConstantCeilDiv(OpBuilder &builder, Location loc,
                                 Type vregType, TranslationContext &ctx,
                                 Value numerator, int64_t divisor) {
  int64_t bias = divisor - 1;
  auto biasImm = ctx.createImmType(bias);
  Value biasConst = ConstantOp::create(builder, loc, biasImm, bias);
  Value biased = V_ADD_U32::create(builder, loc, vregType, biasConst, numerator);
  return emitConstantUnsignedDiv(builder, loc, vregType, ctx, biased, divisor);
}

// Runtime unsigned division via float reciprocal with two-step fixup (VALU).
static Value emitRuntimeUnsignedDiv(OpBuilder &builder, Location loc,
                                    Type vregType, TranslationContext &ctx,
                                    Value numerator, Value divisor) {
  Value fNum = V_CVT_F32_U32::create(builder, loc, vregType, numerator);
  Value fDiv = V_CVT_F32_U32::create(builder, loc, vregType, divisor);
  Value rcp = V_RCP_F32::create(builder, loc, vregType, fDiv);
  Value fQuot = V_MUL_F32::create(builder, loc, vregType, fNum, rcp);
  Value quot = V_CVT_U32_F32::create(builder, loc, vregType, fQuot);

  auto sregType2 = ctx.createSRegType(2, 2);
  auto oneImm = ctx.createImmType(1);
  Value oneConst = ConstantOp::create(builder, loc, oneImm, 1);
  auto zeroImm = ctx.createImmType(0);
  Value zeroConst = ConstantOp::create(builder, loc, zeroImm, 0);

  Value prod = V_MUL_LO_U32::create(builder, loc, vregType, quot, divisor);
  V_CMP_GT_U32::create(builder, loc, prod, numerator);
  Value vcc1 = PrecoloredSRegOp::create(builder, loc, sregType2, 106, 2);
  Value correction = V_CNDMASK_B32::create(builder, loc, vregType, zeroConst, oneConst, vcc1);
  Value fixedQuot = V_SUB_U32::create(builder, loc, vregType, quot, correction);

  Value prod2 = V_MUL_LO_U32::create(builder, loc, vregType, fixedQuot, divisor);
  Value rem = V_SUB_U32::create(builder, loc, vregType, numerator, prod2);
  V_CMP_GE_U32::create(builder, loc, rem, divisor);
  Value vcc2 = PrecoloredSRegOp::create(builder, loc, sregType2, 106, 2);
  Value correction2 = V_CNDMASK_B32::create(builder, loc, vregType, zeroConst, oneConst, vcc2);
  return V_ADD_U32::create(builder, loc, vregType, fixedQuot, correction2);
}

static Value emitRuntimeUnsignedMod(OpBuilder &builder, Location loc,
                                    Type vregType, TranslationContext &ctx,
                                    Value numerator, Value divisor) {
  Value quotient = emitRuntimeUnsignedDiv(builder, loc, vregType, ctx, numerator, divisor);
  Value product = V_MUL_LO_U32::create(builder, loc, vregType, quotient, divisor);
  return V_SUB_U32::create(builder, loc, vregType, numerator, product);
}

// Check if all leaf operands (dims/symbols) of an affine expression are scalar.
// If true, the entire expression can be computed on SALU.
static bool isExprFullyScalar(AffineExpr e, affine::AffineApplyOp applyOp,
                              AffineMap map, TranslationContext &ctx) {
  if (isa<AffineConstantExpr>(e))
    return true;
  if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
    if (dimExpr.getPosition() < applyOp.getOperands().size()) {
      Value operand = applyOp.getOperands()[dimExpr.getPosition()];
      if (auto mapped = ctx.getMapper().getMapped(operand)) {
        if (tryGetScalar(*mapped, ctx))
          return true;
      }
    }
    return false;
  }
  if (auto symExpr = dyn_cast<AffineSymbolExpr>(e)) {
    int64_t symIdx = map.getNumDims() + symExpr.getPosition();
    if (symIdx < static_cast<int64_t>(applyOp.getOperands().size())) {
      Value operand = applyOp.getOperands()[symIdx];
      if (auto mapped = ctx.getMapper().getMapped(operand)) {
        if (tryGetScalar(*mapped, ctx))
          return true;
      }
    }
    return false;
  }
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(e)) {
    // FloorDiv, CeilDiv, Mod with non-constant RHS require runtime division.
    // We have no SALU runtime division, so reject these.
    if (binExpr.getKind() == AffineExprKind::FloorDiv ||
        binExpr.getKind() == AffineExprKind::CeilDiv ||
        binExpr.getKind() == AffineExprKind::Mod) {
      if (!isa<AffineConstantExpr>(binExpr.getRHS()))
        return false;
    }
    return isExprFullyScalar(binExpr.getLHS(), applyOp, map, ctx) &&
           isExprFullyScalar(binExpr.getRHS(), applyOp, map, ctx);
  }
  return false;
}

// Check if all leaf operands are scalar (uniform across lanes).
// Unlike isExprFullyScalar, this does NOT reject runtime divisors.
// Used to decide whether the result of a VALU computation can be converted
// back to SGPR via V_READFIRSTLANE_B32.
static bool isExprUniform(AffineExpr e, affine::AffineApplyOp applyOp,
                          AffineMap map, TranslationContext &ctx) {
  if (isa<AffineConstantExpr>(e))
    return true;
  if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
    if (dimExpr.getPosition() < applyOp.getOperands().size()) {
      Value operand = applyOp.getOperands()[dimExpr.getPosition()];
      if (auto mapped = ctx.getMapper().getMapped(operand)) {
        if (tryGetScalar(*mapped, ctx))
          return true;
      }
    }
    return false;
  }
  if (auto symExpr = dyn_cast<AffineSymbolExpr>(e)) {
    int64_t symIdx = map.getNumDims() + symExpr.getPosition();
    if (symIdx < static_cast<int64_t>(applyOp.getOperands().size())) {
      Value operand = applyOp.getOperands()[symIdx];
      if (auto mapped = ctx.getMapper().getMapped(operand)) {
        if (tryGetScalar(*mapped, ctx))
          return true;
      }
    }
    return false;
  }
  if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(e)) {
    return isExprUniform(binExpr.getLHS(), applyOp, map, ctx) &&
           isExprUniform(binExpr.getRHS(), applyOp, map, ctx);
  }
  return false;
}

/// Handle affine.apply - compile affine expression to arithmetic instructions
LogicalResult handleAffineApply(Operation *op, TranslationContext &ctx) {
  auto applyOp = cast<affine::AffineApplyOp>(op);
  auto &builder = ctx.getBuilder();
  auto loc = op->getLoc();
  auto vregType = ctx.createVRegType();

  auto map = applyOp.getAffineMap();

  // Get the single operand (for single-dimension maps)
  if (applyOp.getOperands().empty()) {
    return op->emitError("affine.apply with no operands");
  }

  Value baseValue;
  if (auto mapped = ctx.getMapper().getMapped(applyOp.getOperands()[0])) {
    baseValue = *mapped;
  } else {
    return op->emitError("operand not mapped");
  }

  // For single result affine maps, analyze the expression
  if (map.getNumResults() != 1) {
    return op->emitError("only single-result affine maps supported");
  }

  AffineExpr expr = map.getResult(0);

  // Get thread ID upper bound for the first operand (used for simplification)
  // If the first operand is a thread ID with known upper bound, we can
  // simplify floor divisions where divisor >= upper_bound to 0
  int64_t threadIdUpperBound = 0;
  if (applyOp.getOperands().size() > 0) {
    threadIdUpperBound = ctx.getThreadIdUpperBound(applyOp.getOperands()[0]);
  }

  // HIGH-LEVEL SIMPLIFICATION: Check if the entire expression simplifies to
  // just the input symbol when floor divisions evaluate to 0 Pattern: s0 + (s0
  // floordiv N) * C where N >= upper_bound
  //       => s0 + 0 * C = s0
  if (threadIdUpperBound > 0) {
    // Check if expression is Add(symbol, Mul(FloorDiv(symbol, N), C))
    // where N >= threadIdUpperBound
    if (auto addExpr = dyn_cast<AffineBinaryOpExpr>(expr)) {
      if (addExpr.getKind() == AffineExprKind::Add) {
        // Check if LHS is the symbol and RHS is a Mul containing FloorDiv
        if (isa<AffineSymbolExpr>(addExpr.getLHS())) {
          if (auto mulExpr = dyn_cast<AffineBinaryOpExpr>(addExpr.getRHS())) {
            if (mulExpr.getKind() == AffineExprKind::Mul) {
              // Check if LHS of Mul is FloorDiv with divisor >= upperBound
              if (auto floorExpr =
                      dyn_cast<AffineBinaryOpExpr>(mulExpr.getLHS())) {
                if (floorExpr.getKind() == AffineExprKind::FloorDiv) {
                  if (auto constDiv =
                          dyn_cast<AffineConstantExpr>(floorExpr.getRHS())) {
                    if (constDiv.getValue() >= threadIdUpperBound) {
                      // Expression simplifies to just the symbol (s0)
                      // Map result to the thread ID value
                      ctx.getMapper().mapValue(applyOp.getResult(), baseValue);
                      return success();
                    }
                  }
                }
              }
              // Also check RHS of Mul
              if (auto floorExpr =
                      dyn_cast<AffineBinaryOpExpr>(mulExpr.getRHS())) {
                if (floorExpr.getKind() == AffineExprKind::FloorDiv) {
                  if (auto constDiv =
                          dyn_cast<AffineConstantExpr>(floorExpr.getRHS())) {
                    if (constDiv.getValue() >= threadIdUpperBound) {
                      ctx.getMapper().mapValue(applyOp.getResult(), baseValue);
                      return success();
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // NOTE: We used to extract constant addends for buffer store offset:N
  // optimization but this caused bugs when the affine result was used in arith
  // operations (the constant was lost). For now, just compile the full
  // expression.
  // TODO: Re-enable constant extraction only for values used directly in memory
  // ops
  int64_t constAddend = 0;
  AffineExpr exprToCompile = expr;

  // If the entire expression is scalar (all operands are SGPR/precolored),
  // compile it entirely on SALU to avoid VGPR pressure.
  if (isExprFullyScalar(exprToCompile, applyOp, map, ctx)) {
    std::function<Value(AffineExpr)> compileScalar =
        [&](AffineExpr e) -> Value {
      auto sregType = ctx.createSRegType();

      if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
        Value operand = applyOp.getOperands()[dimExpr.getPosition()];
        auto mapped = ctx.getMapper().getMapped(operand);
        return *tryGetScalar(*mapped, ctx);
      }
      if (auto symExpr = dyn_cast<AffineSymbolExpr>(e)) {
        int64_t symIdx = map.getNumDims() + symExpr.getPosition();
        Value operand = applyOp.getOperands()[symIdx];
        auto mapped = ctx.getMapper().getMapped(operand);
        return *tryGetScalar(*mapped, ctx);
      }
      if (auto constExpr = dyn_cast<AffineConstantExpr>(e)) {
        int64_t val = constExpr.getValue();
        auto immType = ctx.createImmType(val);
        Value c = ConstantOp::create(builder, loc, immType, val);
        return static_cast<Value>(S_MOV_B32::create(builder, loc, sregType, c));
      }
      if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(e)) {
        Value lhs = compileScalar(binExpr.getLHS());
        Value rhs = compileScalar(binExpr.getRHS());

        switch (binExpr.getKind()) {
        case AffineExprKind::Add:
          return S_ADD_U32::create(builder, loc, sregType, sregType, lhs, rhs).getDst();
        case AffineExprKind::Mul: {
          if (auto c = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            int64_t val = c.getValue();
            if (val > 0 && isPowerOf2(val)) {
              auto imm = ctx.createImmType(log2(val));
              Value shift = ConstantOp::create(builder, loc, imm, log2(val));
              return S_LSHL_B32::create(builder, loc, sregType, lhs, shift);
            }
          }
          if (auto c = dyn_cast<AffineConstantExpr>(binExpr.getLHS())) {
            int64_t val = c.getValue();
            if (val > 0 && isPowerOf2(val)) {
              auto imm = ctx.createImmType(log2(val));
              Value shift = ConstantOp::create(builder, loc, imm, log2(val));
              return S_LSHL_B32::create(builder, loc, sregType, rhs, shift);
            }
          }
          return S_MUL_I32::create(builder, loc, sregType, lhs, rhs);
        }
        case AffineExprKind::FloorDiv: {
          if (auto c = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            int64_t d = c.getValue();
            if (d > 0 && isPowerOf2(d)) {
              auto imm = ctx.createImmType(log2(d));
              Value shift = ConstantOp::create(builder, loc, imm, log2(d));
              return S_LSHR_B32::create(builder, loc, sregType, lhs, shift);
            }
            return emitScalarConstantUnsignedDiv(builder, loc, ctx, lhs, d);
          }
          // Runtime divisor: fall through to VALU below
          break;
        }
        case AffineExprKind::CeilDiv: {
          if (auto c = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            int64_t d = c.getValue();
            if (d > 0 && isPowerOf2(d)) {
              int64_t bias = d - 1;
              auto biasImm = ctx.createImmType(bias);
              Value biasC = ConstantOp::create(builder, loc, biasImm, bias);
              Value biased = S_ADD_U32::create(builder, loc, sregType, sregType, biasC, lhs).getDst();
              auto imm = ctx.createImmType(log2(d));
              Value shift = ConstantOp::create(builder, loc, imm, log2(d));
              return S_LSHR_B32::create(builder, loc, sregType, biased, shift);
            }
            return emitScalarConstantCeilDiv(builder, loc, ctx, lhs, d);
          }
          break;
        }
        case AffineExprKind::Mod: {
          if (auto c = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            int64_t d = c.getValue();
            if (d > 0 && isPowerOf2(d)) {
              auto maskImm = ctx.createImmType(d - 1);
              Value mask = ConstantOp::create(builder, loc, maskImm, d - 1);
              return S_AND_B32::create(builder, loc, sregType, lhs, mask);
            }
            return emitScalarConstantUnsignedMod(builder, loc, ctx, lhs, d);
          }
          break;
        }
        default:
          break;
        }
      }
      // Should not reach here for a fully-scalar expression with constant RHS.
      // Fallthrough: will fall to VALU path.
      return Value();
    };

    Value result = compileScalar(exprToCompile);
    if (result) {
      ctx.getMapper().mapValue(applyOp.getResult(), result);
      return success();
    }
    // Fall through to VALU if scalar compilation failed
    // (e.g., runtime divisor in a fully-scalar expression)
  }

  // Result type that includes bit range tracking for OR optimization
  struct ExprResult {
    Value value;
    BitRange range;
    ExprResult(Value v, BitRange r) : value(v), range(r) {}
  };

  // Build a string cache key for a sub-expression, incorporating
  // the resolved leaf operand Values so that identical AffineExpr objects
  // with different operand mappings produce different keys.
  std::function<void(AffineExpr, llvm::raw_string_ostream &)> buildCacheKey =
      [&](AffineExpr e, llvm::raw_string_ostream &os) {
    if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
      Value operand = applyOp.getOperands()[dimExpr.getPosition()];
      if (auto mapped = ctx.getMapper().getMapped(operand))
        os << "d" << mapped->getAsOpaquePointer();
      else
        os << "d?";
      return;
    }
    if (auto symExpr = dyn_cast<AffineSymbolExpr>(e)) {
      int64_t symIdx = map.getNumDims() + symExpr.getPosition();
      Value operand = applyOp.getOperands()[symIdx];
      if (auto mapped = ctx.getMapper().getMapped(operand))
        os << "s" << mapped->getAsOpaquePointer();
      else
        os << "s?";
      return;
    }
    if (auto constExpr = dyn_cast<AffineConstantExpr>(e)) {
      os << "c" << constExpr.getValue();
      return;
    }
    if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(e)) {
      os << "(";
      buildCacheKey(binExpr.getLHS(), os);
      os << static_cast<int>(binExpr.getKind());
      buildCacheKey(binExpr.getRHS(), os);
      os << ")";
      return;
    }
  };

  auto &subExprCache = ctx.affineSubExprCache;

  // Inner compilation function (does the real work)
  std::function<ExprResult(AffineExpr)> compileExprInner;

  // Check if an expression contains a runtime (non-constant) divisor.
  // Only these expensive sub-trees benefit from caching.
  std::function<bool(AffineExpr)> hasRuntimeDiv =
      [&](AffineExpr e) -> bool {
    if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(e)) {
      if (binExpr.getKind() == AffineExprKind::FloorDiv ||
          binExpr.getKind() == AffineExprKind::CeilDiv ||
          binExpr.getKind() == AffineExprKind::Mod) {
        if (!isa<AffineConstantExpr>(binExpr.getRHS()))
          return true;
      }
      return hasRuntimeDiv(binExpr.getLHS()) || hasRuntimeDiv(binExpr.getRHS());
    }
    return false;
  };

  // Memoizing wrapper: check cache, compile if needed, store result
  std::function<ExprResult(AffineExpr)> compileExpr =
      [&](AffineExpr e) -> ExprResult {
    // Only cache binary sub-expressions that contain runtime division.
    // Cheap expressions (shifts, masks, adds) are cheaper to recompute
    // than to keep their results alive across long ranges.
    if (!isa<AffineBinaryOpExpr>(e) || !hasRuntimeDiv(e))
      return compileExprInner(e);

    std::string cacheKey;
    llvm::raw_string_ostream keyOs(cacheKey);
    buildCacheKey(e, keyOs);

    auto cacheIt = subExprCache.find(cacheKey);
    if (cacheIt != subExprCache.end()) {
      auto &entry = cacheIt->second;
      Value cached = Value::getFromOpaquePointer(entry.valuePtr);
      if (cached && cached.getDefiningOp()) {
        return ExprResult(cached, BitRange(entry.rangeLow, entry.rangeHigh));
      }
      subExprCache.erase(cacheIt);
    }

    ExprResult result = compileExprInner(e);
    if (result.value) {
      TranslationContext::CachedSubExpr entry;
      entry.valuePtr = result.value.getAsOpaquePointer();
      entry.rangeLow = result.range.lowBit;
      entry.rangeHigh = result.range.highBit;
      subExprCache[cacheKey] = entry;
    }
    return result;
  };

  compileExprInner =
      [&](AffineExpr e) -> ExprResult {
    // Dimension reference
    if (auto dimExpr = dyn_cast<AffineDimExpr>(e)) {
      if (dimExpr.getPosition() < applyOp.getOperands().size()) {
        Value operand = applyOp.getOperands()[dimExpr.getPosition()];
        if (auto mapped = ctx.getMapper().getMapped(operand)) {
          // Use tracked bit range if available
          BitRange range = ctx.getBitRange(*mapped);
          return ExprResult(*mapped, range);
        }
      }
      return ExprResult(baseValue, ctx.getBitRange(baseValue));
    }

    // Symbol reference
    if (auto symExpr = dyn_cast<AffineSymbolExpr>(e)) {
      int64_t symIdx = map.getNumDims() + symExpr.getPosition();
      if (symIdx < static_cast<int64_t>(applyOp.getOperands().size())) {
        Value operand = applyOp.getOperands()[symIdx];
        if (auto mapped = ctx.getMapper().getMapped(operand)) {
          BitRange range = ctx.getBitRange(*mapped);
          return ExprResult(*mapped, range);
        }
      }
      return ExprResult(baseValue, ctx.getBitRange(baseValue));
    }

    // Constant
    if (auto constExpr = dyn_cast<AffineConstantExpr>(e)) {
      int64_t val = constExpr.getValue();
      auto immType = ctx.createImmType(val);
      Value constVal = ConstantOp::create(builder, loc, immType, val);
      return ExprResult(constVal, BitRange::fromConstant(val));
    }

    // Binary expressions
    if (auto binExpr = dyn_cast<AffineBinaryOpExpr>(e)) {
      ExprResult lhsResult = compileExpr(binExpr.getLHS());
      ExprResult rhsResult = compileExpr(binExpr.getRHS());
      Value lhs = lhsResult.value;
      Value rhs = rhsResult.value;
      BitRange lhsRange = lhsResult.range;
      BitRange rhsRange = rhsResult.range;

      // --- Partial SALU evaluation ---
      // When both operands resolve to scalar types (SGPR/Imm), use SALU
      // instructions to keep the result as SGPR, avoiding VGPR consumption.
      bool lhsIsScalar = isSGPRType(lhs.getType()) ||
                         isa<ImmType>(lhs.getType());
      bool rhsIsScalar = isSGPRType(rhs.getType()) ||
                         isa<ImmType>(rhs.getType());

      if (lhsIsScalar && rhsIsScalar) {
        auto sregType = ctx.createSRegType();
        auto resolveScalar = [&](Value v) -> Value {
          if (isSGPRType(v.getType()))
            return v;
          if (isa<ImmType>(v.getType()))
            return S_MOV_B32::create(builder, loc, sregType, v);
          return v;
        };

        switch (binExpr.getKind()) {
        case AffineExprKind::Add: {
          Value sL = resolveScalar(lhs);
          Value sR = resolveScalar(rhs);
          Value res = S_ADD_U32::create(builder, loc, sregType, sregType,
                                        sL, sR).getDst();
          return ExprResult(res, lhsRange.extendForAdd(rhsRange));
        }
        case AffineExprKind::FloorDiv: {
          if (auto c = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            int64_t d = c.getValue();
            if (d > 0 && isPowerOf2(d)) {
              Value sL = resolveScalar(lhs);
              auto imm = ctx.createImmType(log2(d));
              Value shift = ConstantOp::create(builder, loc, imm, log2(d));
              Value res = S_LSHR_B32::create(builder, loc, sregType,
                                             sL, shift);
              BitRange rr = lhsRange.shiftRight(log2(d));
              ctx.setBitRange(res, rr);
              return ExprResult(res, rr);
            }
            Value sL = resolveScalar(lhs);
            return ExprResult(
                emitScalarConstantUnsignedDiv(builder, loc, ctx, sL, d),
                BitRange());
          }
          break;
        }
        case AffineExprKind::CeilDiv: {
          if (auto c = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            int64_t d = c.getValue();
            if (d > 0 && isPowerOf2(d)) {
              Value sL = resolveScalar(lhs);
              int64_t bias = d - 1;
              auto biasImm = ctx.createImmType(bias);
              Value biasC = ConstantOp::create(builder, loc, biasImm, bias);
              Value biased = S_ADD_U32::create(builder, loc, sregType,
                                               sregType, sL, biasC).getDst();
              auto imm = ctx.createImmType(log2(d));
              Value shift = ConstantOp::create(builder, loc, imm, log2(d));
              return ExprResult(
                  S_LSHR_B32::create(builder, loc, sregType, biased, shift),
                  BitRange());
            }
            Value sL = resolveScalar(lhs);
            return ExprResult(
                emitScalarConstantCeilDiv(builder, loc, ctx, sL, d),
                BitRange());
          }
          break;
        }
        case AffineExprKind::Mod: {
          if (auto c = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
            int64_t d = c.getValue();
            if (d > 0 && isPowerOf2(d)) {
              Value sL = resolveScalar(lhs);
              auto maskImm = ctx.createImmType(d - 1);
              Value mask = ConstantOp::create(builder, loc, maskImm, d - 1);
              Value res = S_AND_B32::create(builder, loc, sregType, sL, mask);
              BitRange rr = BitRange(0, log2(d) - 1);
              ctx.setBitRange(res, rr);
              return ExprResult(res, rr);
            }
            Value sL = resolveScalar(lhs);
            return ExprResult(
                emitScalarConstantUnsignedMod(builder, loc, ctx, sL, d),
                BitRange());
          }
          break;
        }
        default:
          break;
        }
      }

      switch (binExpr.getKind()) {
      case AffineExprKind::Add: {
        if (!lhsRange.overlaps(rhsRange)) {
          auto tryFuseShiftOr =
              [&](AffineExpr shiftExpr, Value orend,
                  BitRange orendRange) -> std::optional<ExprResult> {
            if (auto mulExpr = dyn_cast<AffineBinaryOpExpr>(shiftExpr)) {
              if (mulExpr.getKind() == AffineExprKind::Mul) {
                if (auto constRhs =
                        dyn_cast<AffineConstantExpr>(mulExpr.getRHS())) {
                  int64_t val = constRhs.getValue();
                  if (val > 0 && (val & (val - 1)) == 0) {
                    int64_t shiftAmount = log2(val);
                    ExprResult baseResult = compileExpr(mulExpr.getLHS());
                    auto shiftImm = ctx.createImmType(shiftAmount);
                    auto shiftConst =
                        ConstantOp::create(builder, loc, shiftImm, shiftAmount);
                    Value fusedResult = V_LSHL_OR_B32::create(
                        builder, loc, vregType, baseResult.value, shiftConst,
                        orend);
                    BitRange shiftedRange =
                        baseResult.range.shiftLeft(shiftAmount);
                    BitRange resultRange = shiftedRange.merge(orendRange);
                    ctx.setBitRange(fusedResult, resultRange);
                    return ExprResult(fusedResult, resultRange);
                  }
                }
                if (auto constLhs =
                        dyn_cast<AffineConstantExpr>(mulExpr.getLHS())) {
                  int64_t val = constLhs.getValue();
                  if (val > 0 && (val & (val - 1)) == 0) {
                    int64_t shiftAmount = log2(val);
                    ExprResult baseResult = compileExpr(mulExpr.getRHS());
                    auto shiftImm = ctx.createImmType(shiftAmount);
                    auto shiftConst =
                        ConstantOp::create(builder, loc, shiftImm, shiftAmount);
                    Value fusedResult = V_LSHL_OR_B32::create(
                        builder, loc, vregType, baseResult.value, shiftConst,
                        orend);
                    BitRange shiftedRange =
                        baseResult.range.shiftLeft(shiftAmount);
                    BitRange resultRange = shiftedRange.merge(orendRange);
                    ctx.setBitRange(fusedResult, resultRange);
                    return ExprResult(fusedResult, resultRange);
                  }
                }
              }
            }
            return std::nullopt;
          };

          if (auto result = tryFuseShiftOr(binExpr.getLHS(), rhs, rhsRange)) {
            return *result;
          }
          if (auto result = tryFuseShiftOr(binExpr.getRHS(), lhs, lhsRange)) {
            return *result;
          }

          Value orResult = V_OR_B32::create(builder, loc, vregType, lhs, rhs);
          BitRange resultRange = lhsRange.merge(rhsRange);
          ctx.setBitRange(orResult, resultRange);
          return ExprResult(orResult, resultRange);
        }
        // Overlapping ranges - must use ADD
        Value addResult = V_ADD_U32::create(builder, loc, vregType, lhs, rhs);
        BitRange resultRange = lhsRange.extendForAdd(rhsRange);
        ctx.setBitRange(addResult, resultRange);
        return ExprResult(addResult, resultRange);
      }

      case AffineExprKind::Mul: {
        // PATTERN: floor(x / N) * N = x & ~(N-1)  when N is power of 2
        // Detect Mul(FloorDiv(expr, N), N) and emit AND directly.
        // This saves 1 instruction vs (x >> log2(N)) << log2(N).
        auto tryFloorMulToAnd =
            [&](AffineExpr divExpr,
                AffineExpr constExpr) -> std::optional<ExprResult> {
          auto floorDiv = dyn_cast<AffineBinaryOpExpr>(divExpr);
          if (!floorDiv || floorDiv.getKind() != AffineExprKind::FloorDiv)
            return std::nullopt;
          auto mulConst = dyn_cast<AffineConstantExpr>(constExpr);
          auto divConst = dyn_cast<AffineConstantExpr>(floorDiv.getRHS());
          if (!mulConst || !divConst)
            return std::nullopt;
          if (mulConst.getValue() != divConst.getValue())
            return std::nullopt;
          int64_t N = mulConst.getValue();
          if (N <= 0 || !isPowerOf2(N))
            return std::nullopt;
          // Compile the inner expression (the x in floor(x/N)*N)
          ExprResult innerResult = compileExpr(floorDiv.getLHS());
          int64_t mask = ~(N - 1) & 0xFFFFFFFF;
          auto maskImm = ctx.createImmType(mask);
          auto maskConst = ConstantOp::create(builder, loc, maskImm, mask);
          // NOTE: constant must be src0 (first operand) for VOP2 encoding.
          // src1 must be a VGPR on AMDGCN.
          Value andResult = V_AND_B32::create(builder, loc, vregType, maskConst,
                                              innerResult.value);
          // Result has same bit range as inner, but low bits cleared
          BitRange resultRange = innerResult.range;
          // Clear bits below log2(N) -- conservative: use inner range
          ctx.setBitRange(andResult, resultRange);
          return ExprResult(andResult, resultRange);
        };

        if (auto result = tryFloorMulToAnd(binExpr.getLHS(), binExpr.getRHS()))
          return *result;
        if (auto result = tryFloorMulToAnd(binExpr.getRHS(), binExpr.getLHS()))
          return *result;

        // Constant folding: if either operand is constant 0, result is 0
        if (auto constLhs = dyn_cast<AffineConstantExpr>(binExpr.getLHS())) {
          if (constLhs.getValue() == 0) {
            auto immZero = ctx.createImmType(0);
            return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                              BitRange(0, 0));
          }
        }
        if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          if (constRhs.getValue() == 0) {
            auto immZero = ctx.createImmType(0);
            return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                              BitRange(0, 0));
          }
          int64_t val = constRhs.getValue();
          if (isPowerOf2(val)) {
            int64_t shiftAmount = log2(val);
            auto shiftAmt = ctx.createImmType(shiftAmount);
            auto shiftConst =
                ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
            Value shiftResult =
                V_LSHLREV_B32::create(builder, loc, vregType, shiftConst, lhs);
            BitRange resultRange = lhsRange.shiftLeft(shiftAmount);
            ctx.setBitRange(shiftResult, resultRange);
            return ExprResult(shiftResult, resultRange);
          }
        }
        if (auto constLhs = dyn_cast<AffineConstantExpr>(binExpr.getLHS())) {
          int64_t val = constLhs.getValue();
          if (isPowerOf2(val)) {
            int64_t shiftAmount = log2(val);
            auto shiftAmt = ctx.createImmType(shiftAmount);
            auto shiftConst =
                ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
            Value shiftResult =
                V_LSHLREV_B32::create(builder, loc, vregType, shiftConst, rhs);
            BitRange resultRange = rhsRange.shiftLeft(shiftAmount);
            ctx.setBitRange(shiftResult, resultRange);
            return ExprResult(shiftResult, resultRange);
          }
        }
        Value mulResult =
            V_MUL_LO_U32::create(builder, loc, vregType, lhs, rhs);
        return ExprResult(mulResult, BitRange());
      }

      case AffineExprKind::FloorDiv: {
        // Check if RHS is constant
        if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          int64_t divisor = constRhs.getValue();

          // If the LHS max value (from BitRange) is less than the divisor,
          // floor(lhs / divisor) == 0 for all inputs. Generalizes the old
          // thread-ID upper-bound check to work with any tracked bit-range.
          if (lhsRange.highBit < 31) {
            int64_t maxVal = (1LL << (lhsRange.highBit + 1)) - 1;
            if (maxVal < divisor) {
              auto immZero = ctx.createImmType(0);
              return ExprResult(ConstantOp::create(builder, loc, immZero, 0),
                                BitRange(0, 0));
            }
          }

          // Check if divisor is power of 2 -> use right shift
          if (isPowerOf2(divisor)) {
            int64_t shiftAmount = log2(divisor);
            auto shiftAmt = ctx.createImmType(shiftAmount);
            auto shiftConst =
                ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
            Value shiftResult =
                V_LSHRREV_B32::create(builder, loc, vregType, shiftConst, lhs);
            BitRange resultRange = lhsRange.shiftRight(shiftAmount);
            ctx.setBitRange(shiftResult, resultRange);
            return ExprResult(shiftResult, resultRange);
          }

          // Non-power-of-2 constant: Granlund-Montgomery
          if (auto sLhs = tryGetScalar(lhs, ctx)) {
            Value divResult = emitScalarConstantUnsignedDiv(builder, loc, ctx,
                                                            *sLhs, divisor);
            return ExprResult(divResult, BitRange());
          }
          Value divResult = emitConstantUnsignedDiv(builder, loc, vregType,
                                                    ctx, lhs, divisor);
          return ExprResult(divResult, BitRange());
        }

        // Runtime divisor: float reciprocal approximation
        Value divResult = emitRuntimeUnsignedDiv(builder, loc, vregType,
                                                 ctx, lhs, rhs);
        return ExprResult(divResult, BitRange());
      }

      case AffineExprKind::CeilDiv: {
        if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          int64_t divisor = constRhs.getValue();

          // Optimization: ceildiv(x, 2^k) = (x + 2^k - 1) >> k.
          // Only valid for non-negative x because V_LSHRREV_B32 is a logical
          // (unsigned) right shift. Negative values would produce a large
          // positive result instead of the correct negative ceiling.
          // Affine expressions in this context originate from index
          // computations which are non-negative by construction.
          if (isPowerOf2(divisor)) {
            int64_t shiftAmount = log2(divisor);
            int64_t bias = divisor - 1;
            auto biasImm = ctx.createImmType(bias);
            auto biasConst = ConstantOp::create(builder, loc, biasImm, bias);
            Value biased =
                V_ADD_U32::create(builder, loc, vregType, biasConst, lhs);
            auto shiftAmt = ctx.createImmType(shiftAmount);
            auto shiftConst =
                ConstantOp::create(builder, loc, shiftAmt, shiftAmount);
            Value shiftResult = V_LSHRREV_B32::create(builder, loc, vregType,
                                                      shiftConst, biased);
            BitRange resultRange =
                lhsRange.extendForAdd(BitRange::fromConstant(bias))
                    .shiftRight(shiftAmount);
            ctx.setBitRange(shiftResult, resultRange);
            return ExprResult(shiftResult, resultRange);
          }

          if (auto sLhs = tryGetScalar(lhs, ctx)) {
            Value ceilResult = emitScalarConstantCeilDiv(builder, loc, ctx,
                                                         *sLhs, divisor);
            return ExprResult(ceilResult, BitRange());
          }
          Value ceilResult = emitConstantCeilDiv(builder, loc, vregType,
                                                 ctx, lhs, divisor);
          return ExprResult(ceilResult, BitRange());
        }

        // Runtime ceildiv: (lhs + rhs - 1) / rhs
        {
          auto oneImm = ctx.createImmType(1);
          Value oneConst = ConstantOp::create(builder, loc, oneImm, 1);
          Value rhsM1 = V_SUB_U32::create(builder, loc, vregType, rhs, oneConst);
          Value biased = V_ADD_U32::create(builder, loc, vregType, lhs, rhsM1);
          Value ceilResult = emitRuntimeUnsignedDiv(builder, loc, vregType,
                                                    ctx, biased, rhs);
          return ExprResult(ceilResult, BitRange());
        }
      }

      case AffineExprKind::Mod: {
        // Check if RHS is constant power of 2 -> use AND
        if (auto constRhs = dyn_cast<AffineConstantExpr>(binExpr.getRHS())) {
          int64_t val = constRhs.getValue();
          if (isPowerOf2(val)) {
            auto maskVal = ctx.createImmType(val - 1);
            auto maskConst = ConstantOp::create(builder, loc, maskVal, val - 1);
            Value andResult =
                V_AND_B32::create(builder, loc, vregType, lhs, maskConst);
            // Result uses bits 0..(log2(val)-1)
            BitRange resultRange = BitRange(0, log2(val) - 1);
            ctx.setBitRange(andResult, resultRange);
            return ExprResult(andResult, resultRange);
          }

          if (auto sLhs = tryGetScalar(lhs, ctx)) {
            Value modResult = emitScalarConstantUnsignedMod(builder, loc, ctx,
                                                            *sLhs, val);
            return ExprResult(modResult, BitRange());
          }
          Value modResult = emitConstantUnsignedMod(builder, loc, vregType,
                                                    ctx, lhs, val);
          return ExprResult(modResult, BitRange());
        }

        // Runtime mod
        Value modResult = emitRuntimeUnsignedMod(builder, loc, vregType,
                                                 ctx, lhs, rhs);
        return ExprResult(modResult, BitRange());
      }

      default:
        return ExprResult(lhs,
                          BitRange()); // Unsupported, return LHS as fallback
      }
    }

    return ExprResult(baseValue, BitRange()); // Fallback
  };

  ExprResult result = compileExpr(exprToCompile);
  Value resultValue = result.value;

  // NOTE: Converting uniform VGPR results to SGPR via V_READFIRSTLANE_B32
  // would reduce VGPR pressure, but the resulting SGPR values may violate
  // AMDGCN encoding constraints when used as VALU operands (constant bus
  // restrictions, src1 must be VGPR for VOP2).  Disabled for now.

  ctx.getMapper().mapValue(applyOp.getResult(), resultValue);
  ctx.setBitRange(resultValue, result.range);

  // Track the constant addend for buffer store offset:N optimization
  if (constAddend != 0) {
    ctx.setConstOffset(applyOp.getResult(), constAddend);
  }

  return success();
}

} // namespace waveasm
