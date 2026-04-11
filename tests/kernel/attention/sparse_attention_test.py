# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import torch
from torch.testing import assert_close

from wave_lang.kernel.wave.templates.sparse_attention_utils import (
    dense_block_pattern,
    causal_block_pattern,
    local_window_block_pattern,
    block_pattern_with_global_tokens,
    dense_mask_to_block_sparse,
    block_sparse_to_dense_mask,
)
from wave_lang.kernel.wave.utils.reference_kernel_utils import (
    sparse_scaled_dot_product_attention,
)
from wave_lang.kernel.wave.templates.sparse_attention import (
    prepare_sparse_attention_inputs,
    get_sparse_bshd_attention_kernel,
)
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
)
from wave_lang.kernel.wave.compile import WaveCompileOptions, wave_compile
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from ..common.utils import (
    param_bool,
    require_e2e,
    require_cdna_2_or_3_or_4,
)

BLOCK_SIZE = 64


# ---------------------------------------------------------------------------
# Phase 0a: Pattern utility tests (CPU-only, no GPU required)
# ---------------------------------------------------------------------------


class TestSparsePatternUtils:
    """Tests for CSR block pattern generation utilities."""

    def test_dense_pattern(self):
        """Full attention: every Q block attends to every KV block.
        block_offsets should be [0, num_kv_blocks, 2*num_kv_blocks, ...].
        block_indices should be [0, 1, 2, ..., num_kv_blocks-1] repeated."""
        num_q_blocks = 4
        num_kv_blocks = 4
        offsets, indices = dense_block_pattern(num_q_blocks, num_kv_blocks)

        expected_offsets = torch.tensor(
            [i * num_kv_blocks for i in range(num_q_blocks + 1)], dtype=torch.int32
        )
        assert_close(offsets, expected_offsets)

        expected_indices = torch.arange(num_kv_blocks, dtype=torch.int32).repeat(
            num_q_blocks
        )
        assert_close(indices, expected_indices)

    def test_dense_pattern_rectangular(self):
        """Dense pattern with more KV blocks than Q blocks."""
        num_q_blocks = 2
        num_kv_blocks = 6
        offsets, indices = dense_block_pattern(num_q_blocks, num_kv_blocks)

        assert offsets.shape == (num_q_blocks + 1,)
        assert indices.shape == (num_q_blocks * num_kv_blocks,)
        for i in range(num_q_blocks):
            assert offsets[i + 1] - offsets[i] == num_kv_blocks

    def test_causal_pattern(self):
        """Causal: Q block i attends to KV blocks 0..i.
        block_offsets[i+1] - block_offsets[i] == i+1.
        block_indices for Q block i == [0, 1, ..., i]."""
        num_q_blocks = 4
        num_kv_blocks = 4
        offsets, indices = causal_block_pattern(num_q_blocks, num_kv_blocks)

        for i in range(num_q_blocks):
            row_len = offsets[i + 1] - offsets[i]
            assert row_len == i + 1, f"Q block {i} should attend to {i + 1} KV blocks"
            row_indices = indices[offsets[i] : offsets[i + 1]]
            expected = torch.arange(i + 1, dtype=torch.int32)
            assert_close(row_indices, expected)

    def test_causal_pattern_rectangular(self):
        """Causal pattern with more KV blocks than Q blocks.
        Q block i attends to KV blocks 0..min(i, num_kv_blocks-1)."""
        num_q_blocks = 3
        num_kv_blocks = 6
        offsets, indices = causal_block_pattern(num_q_blocks, num_kv_blocks)

        for i in range(num_q_blocks):
            row_len = offsets[i + 1] - offsets[i]
            assert row_len == i + 1

    def test_local_window_pattern(self):
        """Local window of w blocks: Q block i attends to
        KV blocks max(0, i-w+1)..i.
        Verify block counts and indices."""
        num_q_blocks = 8
        num_kv_blocks = 8
        window_blocks = 3
        offsets, indices = local_window_block_pattern(
            num_q_blocks, num_kv_blocks, window_blocks, causal=True
        )

        for i in range(num_q_blocks):
            start = max(0, i - window_blocks + 1)
            end = i
            expected_len = end - start + 1
            row_len = (offsets[i + 1] - offsets[i]).item()
            assert (
                row_len == expected_len
            ), f"Q block {i}: expected {expected_len} KV blocks, got {row_len}"
            row_indices = indices[offsets[i] : offsets[i + 1]]
            expected = torch.arange(start, end + 1, dtype=torch.int32)
            assert_close(row_indices, expected)

    def test_local_window_pattern_noncausal(self):
        """Non-causal local window: Q block i attends to
        max(0, i-w//2)..min(num_kv-1, i+w//2)."""
        num_q_blocks = 8
        num_kv_blocks = 8
        window_blocks = 3
        offsets, indices = local_window_block_pattern(
            num_q_blocks, num_kv_blocks, window_blocks, causal=False
        )

        for i in range(num_q_blocks):
            start = max(0, i - window_blocks // 2)
            end = min(num_kv_blocks - 1, i + window_blocks // 2)
            expected_len = end - start + 1
            row_len = (offsets[i + 1] - offsets[i]).item()
            assert (
                row_len == expected_len
            ), f"Q block {i}: expected {expected_len} KV blocks, got {row_len}"

    def test_causal_local_window_pattern(self):
        """Combined causal + local window: Q block i attends to
        KV blocks max(0, i-w+1)..i (same as local window when
        causal, since causal already restricts to <=i)."""
        num_q_blocks = 8
        num_kv_blocks = 8
        window_blocks = 2
        offsets, indices = local_window_block_pattern(
            num_q_blocks, num_kv_blocks, window_blocks, causal=True
        )

        for i in range(num_q_blocks):
            start = max(0, i - window_blocks + 1)
            end = i
            expected_len = end - start + 1
            row_len = (offsets[i + 1] - offsets[i]).item()
            assert row_len == expected_len

    def test_block_sparse_from_mask(self):
        """Given an arbitrary [num_q_blocks, num_kv_blocks] bool tensor,
        convert to CSR.  Round-trip: CSR back to dense should match."""
        num_q_blocks = 4
        num_kv_blocks = 6
        torch.manual_seed(42)
        mask = torch.rand(num_q_blocks, num_kv_blocks) > 0.5
        offsets, indices = dense_mask_to_block_sparse(mask)
        roundtrip = block_sparse_to_dense_mask(
            offsets, indices, num_q_blocks, num_kv_blocks
        )
        assert_close(roundtrip, mask)

    def test_block_sparse_from_mask_all_true(self):
        """All-true mask should produce dense pattern."""
        num_q_blocks = 3
        num_kv_blocks = 4
        mask = torch.ones(num_q_blocks, num_kv_blocks, dtype=torch.bool)
        offsets, indices = dense_mask_to_block_sparse(mask)
        assert indices.shape[0] == num_q_blocks * num_kv_blocks

    def test_block_sparse_from_mask_all_false(self):
        """All-false mask should produce empty pattern."""
        num_q_blocks = 3
        num_kv_blocks = 4
        mask = torch.zeros(num_q_blocks, num_kv_blocks, dtype=torch.bool)
        offsets, indices = dense_mask_to_block_sparse(mask)
        assert indices.shape[0] == 0
        assert (offsets == 0).all()

    def test_pattern_with_global_tokens(self):
        """Longformer-style: local window + first G KV blocks always
        attended.  Verify the union is correct."""
        num_q_blocks = 8
        num_kv_blocks = 8
        window_blocks = 2
        num_global = 2

        base_offsets, base_indices = local_window_block_pattern(
            num_q_blocks, num_kv_blocks, window_blocks, causal=True
        )
        offsets, indices = block_pattern_with_global_tokens(
            base_offsets,
            base_indices,
            num_q_blocks,
            num_kv_blocks,
            num_global,
        )

        result_mask = block_sparse_to_dense_mask(
            offsets, indices, num_q_blocks, num_kv_blocks
        )
        # First num_global KV blocks should be attended by every Q block
        assert result_mask[:, :num_global].all()
        # The local window pattern should also be present
        base_mask = block_sparse_to_dense_mask(
            base_offsets, base_indices, num_q_blocks, num_kv_blocks
        )
        # Union: result should be superset of base
        assert (result_mask | ~base_mask).all()


# ---------------------------------------------------------------------------
# Phase 0b: Reference implementation tests (CPU-only)
# ---------------------------------------------------------------------------


class TestSparseAttentionReference:
    """Tests for the sparse attention reference implementation."""

    def test_reference_matches_pytorch_dense(self):
        """With a fully-dense pattern, sparse reference must match
        torch.nn.functional.scaled_dot_product_attention."""
        torch.manual_seed(42)
        B, H, S_q, S_kv, D = 1, 4, 128, 128, 64
        block_size = 64
        num_q_blocks = S_q // block_size
        num_kv_blocks = S_kv // block_size

        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, S_kv, D)
        v = torch.randn(B, H, S_kv, D)

        offsets, indices = dense_block_pattern(num_q_blocks, num_kv_blocks)

        ref = sparse_scaled_dot_product_attention(q, k, v, offsets, indices, block_size)
        torch_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        assert_close(ref, torch_ref, atol=1e-5, rtol=1e-5)

    def test_reference_matches_pytorch_causal(self):
        """With a causal block pattern, sparse reference must match
        torch SDPA with is_causal=True (up to block granularity
        differences at the diagonal)."""
        torch.manual_seed(42)
        B, H, S_q, S_kv, D = 1, 4, 256, 256, 64
        block_size = 64
        num_q_blocks = S_q // block_size
        num_kv_blocks = S_kv // block_size

        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, S_kv, D)
        v = torch.randn(B, H, S_kv, D)

        offsets, indices = causal_block_pattern(num_q_blocks, num_kv_blocks)

        ref = sparse_scaled_dot_product_attention(q, k, v, offsets, indices, block_size)

        # The block-level causal mask is coarser than element-level causal.
        # Within each diagonal block, the block-sparse reference attends to all
        # elements (no element-level causal masking within the block).
        # So we compare against PyTorch SDPA with a block-level causal mask.
        mask = block_sparse_to_dense_mask(offsets, indices, num_q_blocks, num_kv_blocks)
        # Expand to element level
        elem_mask = mask.repeat_interleave(block_size, dim=0).repeat_interleave(
            block_size, dim=1
        )
        elem_mask = elem_mask.unsqueeze(0).unsqueeze(0).expand(B, H, S_q, S_kv)
        attn_mask = torch.where(elem_mask, 0.0, float("-inf"))
        torch_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )

        assert_close(ref, torch_ref, atol=1e-5, rtol=1e-5)

    def test_reference_local_window(self):
        """With local window pattern, verify output matches manual
        computation on a small (4 block x 4 block) example."""
        torch.manual_seed(42)
        B, H, D = 1, 2, 32
        block_size = 16
        num_q_blocks = 4
        num_kv_blocks = 4
        S_q = num_q_blocks * block_size
        S_kv = num_kv_blocks * block_size
        window_blocks = 2

        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, S_kv, D)
        v = torch.randn(B, H, S_kv, D)

        offsets, indices = local_window_block_pattern(
            num_q_blocks, num_kv_blocks, window_blocks, causal=True
        )

        ref = sparse_scaled_dot_product_attention(q, k, v, offsets, indices, block_size)

        # Build element-level mask from block pattern
        mask = block_sparse_to_dense_mask(offsets, indices, num_q_blocks, num_kv_blocks)
        elem_mask = mask.repeat_interleave(block_size, dim=0).repeat_interleave(
            block_size, dim=1
        )
        elem_mask = elem_mask.unsqueeze(0).unsqueeze(0).expand(B, H, S_q, S_kv)
        attn_mask = torch.where(elem_mask, 0.0, float("-inf"))
        torch_ref = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask
        )

        assert_close(ref, torch_ref, atol=1e-5, rtol=1e-5)

    def test_reference_all_masked_row(self):
        """If a Q block has zero KV blocks (empty row in CSR),
        output should be zeros (NaN-safe)."""
        torch.manual_seed(42)
        B, H, D = 1, 2, 32
        block_size = 16
        num_q_blocks = 3
        num_kv_blocks = 3
        S_q = num_q_blocks * block_size
        S_kv = num_kv_blocks * block_size

        q = torch.randn(B, H, S_q, D)
        k = torch.randn(B, H, S_kv, D)
        v = torch.randn(B, H, S_kv, D)

        # Create a pattern where Q block 1 has no KV blocks
        mask = torch.ones(num_q_blocks, num_kv_blocks, dtype=torch.bool)
        mask[1, :] = False
        offsets, indices = dense_mask_to_block_sparse(mask)

        ref = sparse_scaled_dot_product_attention(q, k, v, offsets, indices, block_size)

        # Row 1 (tokens block_size:2*block_size) should be zeros
        assert (ref[:, :, block_size : 2 * block_size, :] == 0).all()
        # Other rows should be non-zero
        assert (ref[:, :, :block_size, :] != 0).any()
        assert (ref[:, :, 2 * block_size :, :] != 0).any()


# ---------------------------------------------------------------------------
# Phase 0c: End-to-end kernel tests (GPU required)
# ---------------------------------------------------------------------------

# (num_heads, query_seq_len, head_size_kv, head_size, kv_seq_len)
sparse_attention_shapes = [
    (8, 256, 64, 64, 256),
    (8, 512, 64, 64, 512),
    (8, 256, 64, 64, 1024),
]


def _run_sparse_attention_kernel(
    input_shape,
    mfma_variant,
    pattern_fn,
    dynamic_dims=False,
):
    """Helper to run sparse attention kernel with a given pattern.

    Args:
        input_shape: (num_heads, query_seq_len, head_size_kv, head_size, kv_seq_len)
        mfma_variant: tuple of two MMATypes
        pattern_fn: callable(num_q_blocks, num_kv_blocks) -> (offsets, indices)
        dynamic_dims: whether to use dynamic dims
    """
    num_heads, query_seq_len, head_size_kv, head_size, kv_seq_len = input_shape
    num_q_blocks = query_seq_len // BLOCK_SIZE
    num_kv_blocks = kv_seq_len // BLOCK_SIZE

    shape = AttentionShape(
        num_query_heads=num_heads,
        num_kv_heads=num_heads,
        query_seq_len=query_seq_len,
        head_size_kv=head_size_kv,
        head_size=head_size,
        kv_seq_len=kv_seq_len,
    )

    offsets, indices = pattern_fn(num_q_blocks, num_kv_blocks)
    kv_token_indices, row_lengths, max_blocks_per_row = prepare_sparse_attention_inputs(
        offsets, indices, num_q_blocks, BLOCK_SIZE, query_seq_len
    )

    kernel_fn, hyperparams, dynamic_symbols = get_sparse_bshd_attention_kernel(
        shape, mfma_variant, max_blocks_per_row, dynamic_dims
    )
    hyperparams.update(get_default_scheduling_params())

    options = WaveCompileOptions(
        subs=hyperparams,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols,
        waves_per_eu=2,
        denorm_fp_math_f32="preserve-sign",
    )
    options = set_default_run_config(options)
    kernel_fn = wave_compile(options, kernel_fn)

    torch.manual_seed(1)
    # BSHD layout: [B, S, H, D]
    q = device_randn(1, query_seq_len, num_heads, head_size, dtype=torch.float16)
    k = device_randn(1, kv_seq_len, num_heads, head_size, dtype=torch.float16)
    v = device_randn(1, kv_seq_len, num_heads, head_size_kv, dtype=torch.float16)
    output = device_zeros(
        1, query_seq_len, num_heads, head_size_kv, dtype=torch.float32
    )

    # Move auxiliary tensors to device
    kv_token_indices_dev = kv_token_indices.to(q.device)
    row_lengths_dev = row_lengths.to(q.device)

    kernel_fn(q, k, v, kv_token_indices_dev, row_lengths_dev, output)

    # Compute reference: transpose to BHSD for the reference function
    q_bhsd = q.squeeze(0).permute(1, 0, 2).unsqueeze(0)  # [1, H, S_q, D]
    k_bhsd = k.squeeze(0).permute(1, 0, 2).unsqueeze(0)  # [1, H, S_kv, D]
    v_bhsd = v.squeeze(0).permute(1, 0, 2).unsqueeze(0)  # [1, H, S_kv, D_v]

    ref = sparse_scaled_dot_product_attention(
        q_bhsd, k_bhsd, v_bhsd, offsets, indices, BLOCK_SIZE
    )
    # ref is [1, H, S_q, D_v] -> transpose to [1, S_q, H, D_v]
    ref_bshd = ref.squeeze(0).permute(1, 0, 2).unsqueeze(0)

    assert_close(output, ref_bshd, check_dtype=False, atol=1e-3, rtol=1e-3)


@require_e2e
class TestSparseAttentionKernel:
    """End-to-end tests for the block-sparse attention Wave kernel."""

    @pytest.mark.parametrize("input_shape", sparse_attention_shapes)
    @pytest.mark.parametrize(
        "mfma_variant",
        [
            pytest.param(
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                marks=require_cdna_2_or_3_or_4,
            ),
        ],
    )
    def test_sparse_attention_dense_pattern(self, input_shape, mfma_variant):
        """With fully-dense CSR pattern, kernel output must match reference."""
        _run_sparse_attention_kernel(input_shape, mfma_variant, dense_block_pattern)

    @pytest.mark.parametrize("input_shape", sparse_attention_shapes)
    @pytest.mark.parametrize(
        "mfma_variant",
        [
            pytest.param(
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                marks=require_cdna_2_or_3_or_4,
            ),
        ],
    )
    def test_sparse_attention_causal_pattern(self, input_shape, mfma_variant):
        """With causal CSR pattern, kernel output must match reference."""
        _run_sparse_attention_kernel(input_shape, mfma_variant, causal_block_pattern)

    @pytest.mark.parametrize("input_shape", sparse_attention_shapes)
    @pytest.mark.parametrize(
        "mfma_variant",
        [
            pytest.param(
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                marks=require_cdna_2_or_3_or_4,
            ),
        ],
    )
    def test_sparse_attention_local_window(self, input_shape, mfma_variant):
        """With local-window CSR pattern, kernel output must match reference."""

        def local_window_fn(num_q, num_kv):
            return local_window_block_pattern(num_q, num_kv, 3, causal=True)

        _run_sparse_attention_kernel(input_shape, mfma_variant, local_window_fn)

    @pytest.mark.parametrize("input_shape", sparse_attention_shapes)
    @pytest.mark.parametrize(
        "mfma_variant",
        [
            pytest.param(
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                marks=require_cdna_2_or_3_or_4,
            ),
        ],
    )
    def test_sparse_attention_mixed_pattern(self, input_shape, mfma_variant):
        """Local window + global tokens pattern, verify against reference."""

        def mixed_fn(num_q, num_kv):
            base_off, base_idx = local_window_block_pattern(
                num_q, num_kv, 2, causal=True
            )
            return block_pattern_with_global_tokens(
                base_off, base_idx, num_q, num_kv, 1
            )

        _run_sparse_attention_kernel(input_shape, mfma_variant, mixed_fn)

    @pytest.mark.parametrize(
        "input_shape",
        [(8, 256, 64, 64, 256)],
    )
    @pytest.mark.parametrize(
        "mfma_variant",
        [
            pytest.param(
                (MMAType.F32_16x16x16_F16, MMAType.F32_16x16x16_F16),
                marks=require_cdna_2_or_3_or_4,
            ),
        ],
    )
    def test_sparse_attention_custom_block_sparse(self, input_shape, mfma_variant):
        """Random block-sparse pattern at ~50% density, verify against reference."""
        num_q_blocks = input_shape[1] // BLOCK_SIZE
        num_kv_blocks = input_shape[4] // BLOCK_SIZE

        def custom_fn(num_q, num_kv):
            torch.manual_seed(42)
            mask = torch.rand(num_q, num_kv) > 0.5
            # Ensure at least one block per row for a valid test
            for i in range(num_q):
                if not mask[i].any():
                    mask[i, 0] = True
            return dense_mask_to_block_sparse(mask)

        _run_sparse_attention_kernel(input_shape, mfma_variant, custom_fn)
