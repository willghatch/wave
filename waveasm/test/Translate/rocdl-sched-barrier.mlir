// RUN: waveasm-translate %s 2>&1 | FileCheck %s
//
// Test: rocdl.sched.barrier is silently dropped (it is an LLVM scheduling
// pseudo-instruction, not a real hardware instruction on AMDGCN).

// CHECK-LABEL: waveasm.program @sched_barrier_test

// The sched.barrier ops should be silently dropped - no raw ops emitted.
// CHECK-NOT: s_sched_barrier

// CHECK: waveasm.s_endpgm

module {
  gpu.module @test_sched_barrier {
    gpu.func @sched_barrier_test() kernel {
      rocdl.sched.barrier 0
      rocdl.sched.barrier 1
      rocdl.sched.barrier 255
      gpu.return
    }
  }
}
