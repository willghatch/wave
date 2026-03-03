// RUN: waveasm-translate --target=gfx942 %s 2>&1 | FileCheck %s

// Test gpu.thread_id translation for single-wave (v_mbcnt) and
// multi-wave (v_bfe_u32 from packed v0) kernels.

module {
  gpu.module @test_kernel {
    // Single-wave: thread_id x uses v_mbcnt pattern
    gpu.func @threadid_x_kernel() kernel {
      // CHECK: waveasm.program @threadid_x_kernel
      // CHECK: waveasm.v_mbcnt_lo_u32_b32
      // CHECK: waveasm.v_mbcnt_hi_u32_b32
      %tid_x = gpu.thread_id x
      gpu.return
    }
  }

  // Multi-wave: thread_id x/y extract from packed v0 using v_bfe_u32.
  // With workgroup_size = (64, 4, 1), amdhsa_system_vgpr_workitem_id = 1
  // so hardware provides packed workitem IDs in v0:
  //   v0[0:9]  = workitem_id_x
  //   v0[10:19] = workitem_id_y
  gpu.module @test_multi_wave {
    gpu.func @multiwave_threadid_kernel() kernel
        attributes {workgroup_size = array<i64: 64, 4, 1>} {
      // CHECK: waveasm.program @multiwave_threadid_kernel
      // CHECK: waveasm.precolored.vreg 0
      // CHECK: waveasm.v_bfe_u32
      %tid_x = gpu.thread_id x
      // CHECK: waveasm.v_bfe_u32
      %tid_y = gpu.thread_id y
      gpu.return
    }
  }
}
