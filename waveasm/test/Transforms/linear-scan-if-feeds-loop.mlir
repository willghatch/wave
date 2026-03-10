// RUN: waveasm-translate --waveasm-linear-scan %s 2>&1 | FileCheck %s
//
// Test: waveasm.if results feeding waveasm.loop init args.
//
// When an if-op result is used as a loop init arg, the register allocator
// ties the if result and the loop block arg to the same physical register.
// The LinearScanPass must set the if-op result type from the allocation
// mapping (not from the then-yield operand type), because the then-yield
// operand may have been allocated to a different physical register (e.g.
// from an inner loop).  Without the fix, the LoopLikeOpInterface verifier
// rejects the mismatch between init arg and region iter_arg types.

// CHECK-LABEL: waveasm.program @if_result_feeds_loop
waveasm.program @if_result_feeds_loop
  target = #waveasm.target<#waveasm.gfx950, 5>
  abi = #waveasm.abi<> {

  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %v0 = waveasm.precolored.vreg 0, 4 : !waveasm.pvreg<0, 4>
  %v4 = waveasm.precolored.vreg 4, 4 : !waveasm.pvreg<4, 4>
  %vs = waveasm.precolored.vreg 8 : !waveasm.pvreg<8>

  %s_zero = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg
  %cmp = waveasm.s_cmp_lt_u32 %s_zero, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg

  // The if produces an AGPR accumulator (then branch) or zero (else branch).
  // The result feeds the loop as an init arg.

  // CHECK: waveasm.if
  %if_result = waveasm.if %cmp : !waveasm.sreg -> !waveasm.areg<4, 4> {
    %acc_init = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.areg<4, 4>
    %mfma = waveasm.v_mfma_scale_f32_16x16x128_f8f6f4 %v0, %v4, %acc_init, %vs, %vs
        : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.areg<4, 4>, !waveasm.pvreg<8>, !waveasm.pvreg<8> -> !waveasm.areg<4, 4>
    waveasm.yield %mfma : !waveasm.areg<4, 4>
  } else {
    %zero_acc = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.areg<4, 4>
    waveasm.yield %zero_acc : !waveasm.areg<4, 4>
  }

  %init_i = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.sreg

  // CHECK: waveasm.loop
  // The loop init arg type must match the block arg type after regalloc.
  // Without the fix, the if result would get the then-yield physical
  // register type (from the MFMA result) while the block arg would get
  // a different physical register, causing a verifier error.
  %i_out, %acc_out = waveasm.loop(%i = %init_i, %acc = %if_result)
      : (!waveasm.sreg, !waveasm.areg<4, 4>) -> (!waveasm.sreg, !waveasm.areg<4, 4>) {

    %new_mfma = waveasm.v_mfma_scale_f32_16x16x128_f8f6f4 %v0, %v4, %acc, %vs, %vs
        : !waveasm.pvreg<0, 4>, !waveasm.pvreg<4, 4>, !waveasm.areg<4, 4>, !waveasm.pvreg<8>, !waveasm.pvreg<8> -> !waveasm.areg<4, 4>

    %next_i:2 = waveasm.s_add_u32 %i, %c1 : !waveasm.sreg, !waveasm.imm<1> -> !waveasm.sreg, !waveasm.sreg
    %loop_cond = waveasm.s_cmp_lt_u32 %next_i#0, %c4 : !waveasm.sreg, !waveasm.imm<4> -> !waveasm.sreg
    waveasm.condition %loop_cond : !waveasm.sreg iter_args(%next_i#0, %new_mfma) : !waveasm.sreg, !waveasm.areg<4, 4>
  }

  waveasm.s_endpgm
}
