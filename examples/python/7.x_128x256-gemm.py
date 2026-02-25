"""
MXFP4 128x256 GEMM matching aiter f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256.

Kernel decisions derived from disassembly of the aiter 128x256 kernel:
  - Tile: M=128, N=256
  - Waves: 2 (M) x 4 (N) = 8 waves per workgroup
  - B is preshuffled (direct global reads, no LDS for B)
  - A goes through LDS (double-buffered)
  - A scale and B scale go directly from global to VGPRs
  - Asymmetric schedule: A triple-buffer (depth-2 prefetch), B from global
  - No split-K

Usage:
    python 7.x_128x256-gemm.py --test test_128x256_preshuffle_b_gemm
    python 7.x_128x256-gemm.py --test test_128x256_preshuffle_b_gemm --debug
"""

import torch

import wave_lang.kernel.lang as tkl
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_preshuffle_b_schedule,
)
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
    b_preshuffle,
    e8m0_shuffle,
)
from utils import parse_args, list_tests, run_test


def _run_mxfp_gemm_preshuffle_b(gemm, shape, out_torch_dtype=torch.float32):
    """Run compiled GEMM kernel with preshuffled B and scales, verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    w_t = w.T.contiguous()
    w_t_ps = b_preshuffle(w_t)
    x_scales_ps = e8m0_shuffle(x_scales)
    w_scales_ps = e8m0_shuffle(w_scales)

    x, w_t_ps = x.cuda(), w_t_ps.cuda()
    x_scales_ps, w_scales_ps = x_scales_ps.cuda(), w_scales_ps.cuda()
    out = torch.zeros(x.shape[0], w_t_ps.shape[0], dtype=out_torch_dtype).cuda()

    gemm(x, x_scales_ps, w_t_ps, w_scales_ps, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def test_128x256_preshuffle_b_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """128x256 MXFP4 GEMM matching aiter BpreShuffle_128x256 kernel (4-wave).

    Uses the K-partitioned preshuffle-B schedule with a 2-stage pipeline:
      - Tile 128x256, 4 waves (1Mx4N)
      - B + B_scale preshuffled (direct global reads, no LDS for B)
      - A through LDS (double-buffered)
      - K-partition interleaving (wave-count-agnostic)
      - Output dtype: f32 (see test_128x256_preshuffle_b_gemm_bf16 for bf16)
    """
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(1, 4)
    )
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.dump_intermediates = "build/intermediates"
    options.dump_binaries = "build/binaries"
    options.print_mlir_file = "gemm_mxfp4_128x256_preshuffle_b.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_preshuffle_b_schedule()

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle_b(gemm, shape)
    print("MXFP 128x256 BpreShuffle GEMM test passed!")


def test_128x256_preshuffle_b_gemm_bf16(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """128x256 MXFP4 GEMM with bf16 output, matching aiter's output dtype.

    Same as test_128x256_preshuffle_b_gemm but outputs bf16 instead of f32.
    The accumulator is still f32; the cast to bf16 happens at the final write.
    This matches aiter's f4gemm_bf16_per1x32Fp4_BpreShuffle_128x256 output.
    """
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(1, 4), output_dtype=tkl.bf16
    )
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.dump_intermediates = "build/intermediates"
    options.dump_binaries = "build/binaries"
    options.print_mlir_file = "gemm_mxfp4_128x256_preshuffle_b_bf16.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_preshuffle_b_schedule()

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle_b(gemm, shape, out_torch_dtype=torch.bfloat16)
    print("MXFP 128x256 BpreShuffle GEMM bf16 output test passed!")


def test_128x256_preshuffle_b_gemm_unroll(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """128x256 MXFP4 GEMM with K-loop unrolling (4-wave).

    Same as test_128x256_preshuffle_b_gemm but with unroll_factor=2.
    The preshuffle-B schedule places all barriers inside clusters so
    tkw.unroll replicates them correctly across unrolled iterations.

    Aiter uses 8x unrolling; this tests the mechanism at 2x first.
    K/BLOCK_K = 8192/256 = 32 iterations; 32/2 = 16 loop iterations with
    step=2, each doing 2 iterations of work.
    """
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(1, 4)
    )
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.dump_intermediates = "build/intermediates"
    options.dump_binaries = "build/binaries"
    options.print_mlir_file = "gemm_mxfp4_128x256_preshuffle_b_unroll.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_preshuffle_b_schedule(unroll_factor=2)

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle_b(gemm, shape)
    print("MXFP 128x256 BpreShuffle GEMM unroll=2 test passed!")


def test_128x256_preshuffle_b_gemm_8wave(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """128x256 MXFP4 GEMM with 8 waves (2Mx4N) matching aiter exactly.

    Uses get_mxfp4_preshuffle_b_schedule which partitions by K (not M),
    making it compatible with wave_shape=(2,4).  The previous asymmetric
    schedule failed with 8 waves because it partitioned by M — with 8 waves
    each wave's M-tile is halved, breaking the interleave parameters.

    Matches aiter decisions:
      - Tile 128x256, 8 waves (2Mx4N) — same as aiter BpreShuffle_128x256
      - B + B_scale preshuffled (direct global reads, no LDS for B)
      - A through LDS (double-buffered)
      - A scale + B scale direct from global
      - K-partitioned schedule (matches aiter's loop structure)
    """
    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, block, wave_shape=(2, 4)
    )
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    options.dump_intermediates = "build/intermediates"
    options.dump_binaries = "build/binaries"
    options.print_mlir_file = "gemm_mxfp4_128x256_preshuffle_b_8wave.mlir"
    options.print_mlir = True
    schedule = get_mxfp4_preshuffle_b_schedule()

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle_b(gemm, shape)
    print("MXFP 128x256 BpreShuffle GEMM 8-wave test passed!")


if __name__ == "__main__":
    args = parse_args()

    if args.list_tests:
        list_tests(globals())
        exit(0)

    if not args.test:
        print("Error: --test argument is required")
        print("Use --list_tests to see available tests")
        exit(1)

    success = run_test(
        args.test, globals(), args.debug, args.repeat, args.shape, args.block
    )
    exit(0 if success else 1)
