# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
MXFP format utilities: input generation, type conversion, and reference kernels.
"""

import torch
from torch import Tensor

from .torch_utils import get_default_device

# HW-specified scale group size for MXFP formats.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(
    shape: tuple[int, int, int], device: torch.device = None
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate random packed MXFP4 inputs and e8m0 scales for scaled GEMM.

    Returns:
        x:        uint8 tensor of shape [M, K//2]   – packed FP4 activations
        w:        uint8 tensor of shape [K//2, N]    – packed FP4 weights (transposed)
        x_scales: uint8 tensor of shape [M, K//32]   – E8M0 activation scales
        w_scales: uint8 tensor of shape [N, K//32]    – E8M0 weight scales

    Note: ``w`` is returned in *transposed* (K//2 x N) layout.  The kernel
    expects N x K//2, so callers must pass ``w.T.contiguous()`` to the kernel.
    ``torchScaledGemmMXFP4`` handles the transpose internally.
    """
    if device is None:
        device = get_default_device()
    M, N, K = shape
    torch.manual_seed(5)
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device=device)
    x = x_low | (x_high << 4)
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device=device)
    w = w_low | (w_high << 4)
    w = w.T
    # Scale range [124, 128) -> exponents {-3, -2, -1, 0} -> scales {0.125, 0.25,
    # 0.5, 1.0}.  Deliberately narrow to keep (FP4 * scale) products within a
    # well-behaved f32 range, avoiding overflow/underflow in the reference path.
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device=device
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device=device
    )
    x_scales = x_scales.T.contiguous()
    w_scales = w_scales.T.contiguous()
    return x, w, x_scales, w_scales


def mxfp4_to_f32(x: Tensor) -> Tensor:
    """Convert packed MXFP4 (e2m1) uint8 values to f32 via lookup table."""
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    lut = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    return torch.tensor(lut, dtype=torch.float32, device=x.device)[x.long()]


def e8m0_to_f32(x: Tensor) -> Tensor:
    """Convert e8m0 scale values to f32.

    E8M0 is an 8-bit exponent-only format: value = 2^(x - 127).
    The special encoding x=255 (all ones) represents NaN.
    """
    nan_mask = x == 255
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[nan_mask] = float("nan")
    return x_f32


def b_preshuffle(b: Tensor) -> Tensor:
    """Preshuffle packed MXFP4 B weights for direct global reads.

    Reorders each 16-row x 32-byte tile from [n, k_sub, k_elem] to
    [k_sub, n, k_elem] so a contiguous 256-byte read fetches one K-chunk
    for all 16 N-rows.

    Input:  [N, K/2] uint8 packed FP4 weights.
    Output: same shape/dtype, rearranged for PRESHUFFLEB.
    """
    N, K_packed = b.shape
    b_5d = b.view(N // 16, 16, K_packed // 32, 2, 16)
    return b_5d.permute(0, 2, 3, 1, 4).contiguous().view(N, K_packed)


def e8m0_shuffle(scale: Tensor) -> Tensor:
    """Shuffle e8m0 scale tensor for hardware preshuffle layout.

    Transforms a [m, n] scale matrix via:
      view(m//32, 2, 16, n//8, 2, 4) -> permute(0,3,5,2,4,1) -> view(m, n)

    See: rocm-libraries PreSwizzle.hpp
    """
    m, n = scale.shape
    sm = (m + 255) // 256 * 256
    sn = (n + 7) // 8 * 8
    padded = torch.zeros(sm, sn, dtype=scale.dtype, device=scale.device)
    padded[:m, :n] = scale
    padded = padded.view(sm // 32, 2, 16, sn // 8, 2, 4)
    padded = padded.permute(0, 3, 5, 2, 4, 1).contiguous()
    return padded.view(sm, sn)[:m, :n].contiguous()


def e8m0_shuffle_tiled(scale: Tensor, n_per_wave: int) -> Tensor:
    """Shuffle e8m0 scale tensor with wave-aligned 32-wide layout.

    Tiles along the first dimension (N) in blocks of n_per_wave, pads
    each tile to the next multiple of 32, then applies the standard
    32-wide e8m0_shuffle within each padded tile.  This ensures every
    wave's scale reads are within its own tile, so the merge pass can
    form DWORDs regardless of N_PER_WAVE alignment.

    Use this instead of e8m0_shuffle when N_PER_WAVE is not a multiple
    of 32.  When N_PER_WAVE is already a multiple of 32 the result is
    identical to e8m0_shuffle (the padding is a no-op).

    Args:
        scale: [N, K_scale] uint8 scale tensor (K_scale = K // 32).
        n_per_wave: Number of N-elements per wave (must be > 0).

    Returns:
        Shuffled tensor of shape [B_SCALE_N, sk] where
        B_SCALE_N = ceil(N/n_per_wave) * ceil32(n_per_wave),
        sk = ceil8(K_scale).
    """
    assert n_per_wave > 0
    N, K_scale = scale.shape
    npw_padded = ((n_per_wave + 31) // 32) * 32
    sk = ((K_scale + 7) // 8) * 8
    n_tiles = (N + n_per_wave - 1) // n_per_wave
    sn = n_tiles * n_per_wave

    padded = torch.zeros(sn, sk, dtype=scale.dtype, device=scale.device)
    padded[:N, :K_scale] = scale

    tiles = padded.view(n_tiles, n_per_wave, sk)

    if npw_padded > n_per_wave:
        pad = torch.zeros(
            n_tiles, npw_padded - n_per_wave, sk,
            dtype=scale.dtype, device=scale.device,
        )
        tiles = torch.cat([tiles, pad], dim=1)

    # Apply standard 32-wide shuffle within each padded tile:
    #   view(n_sub32, 2, 16, sk//8, 2, 4).permute(0, 3, 5, 2, 4, 1)
    # with an extra leading n_tiles dimension.
    tiles = tiles.view(
        n_tiles, npw_padded // 32, 2, 16, sk // 8, 2, 4
    )
    tiles = tiles.permute(0, 1, 4, 6, 3, 5, 2).contiguous()

    return tiles.view(n_tiles * npw_padded, sk)


def torchScaledGemmMXFP4(
    x: Tensor, w: Tensor, x_scales: Tensor, w_scales: Tensor
) -> Tensor:
    """Reference scaled MXFP4 GEMM: dequantize inputs/scales then torch.mm."""
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.T).T
    x_scales_f32 = e8m0_to_f32(
        x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    )
    w_scales_f32 = e8m0_to_f32(
        w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    )
    return torch.mm(x_f32 * x_scales_f32, w_f32 * w_scales_f32.T)
