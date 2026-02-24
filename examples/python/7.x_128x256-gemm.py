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

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
from wave_lang.kernel.wave.schedules import get_mxfp4_asymmetric_schedule
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
    b_preshuffle,
    e8m0_shuffle,
)
from utils import parse_args, list_tests, run_test


def _run_mxfp_gemm_preshuffle_b(gemm, shape):
    """Run compiled GEMM kernel with preshuffled B and scales, verify against reference."""
    x, w, x_scales, w_scales = generate_gemm_afp4wfp4_inputs(shape)
    torch_out = torchScaledGemmMXFP4(x, w, x_scales, w_scales)

    w_t = w.T.contiguous()
    w_t_ps = b_preshuffle(w_t)
    x_scales_ps = e8m0_shuffle(x_scales)
    w_scales_ps = e8m0_shuffle(w_scales)

    x, w_t_ps = x.cuda(), w_t_ps.cuda()
    x_scales_ps, w_scales_ps = x_scales_ps.cuda(), w_scales_ps.cuda()
    out = torch.zeros(x.shape[0], w_t_ps.shape[0], dtype=torch.float32).cuda()

    gemm(x, x_scales_ps, w_t_ps, w_scales_ps, out)
    torch.testing.assert_close(
        torch_out, out.cpu(), check_dtype=False, check_device=False
    )


def test_128x256_preshuffle_b_gemm(
    is_debug=False, shape=(1024, 1024, 8192), block=(128, 256, 256)
):
    """128x256 MXFP4 GEMM matching aiter BpreShuffle_128x256 kernel.

    Matches aiter decisions:
      - Tile 128x256, 8 waves (2Mx4N)
      - B + B_scale preshuffled (direct global reads, no LDS for B)
      - A through LDS (double-buffered / asymmetric prefetch)
      - A scale + B scale direct from global
      - Asymmetric schedule (A: triple-buffer depth-2, B: global-to-VGPR)
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
    schedule = get_mxfp4_asymmetric_schedule()

    options.print_ir_after = "all" if is_debug else []
    options = set_default_run_config(options)
    gemm = wave_compile(options, gemm, schedule)

    _run_mxfp_gemm_preshuffle_b(gemm, shape)
    print("MXFP 128x256 BpreShuffle GEMM test passed!")


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
