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
            assert row_len == expected_len, (
                f"Q block {i}: expected {expected_len} KV blocks, got {row_len}"
            )

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

        ref = sparse_scaled_dot_product_attention(
            q, k, v, offsets, indices, block_size
        )
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

        ref = sparse_scaled_dot_product_attention(
            q, k, v, offsets, indices, block_size
        )

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

        ref = sparse_scaled_dot_product_attention(
            q, k, v, offsets, indices, block_size
        )

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

        ref = sparse_scaled_dot_product_attention(
            q, k, v, offsets, indices, block_size
        )

        # Row 1 (tokens block_size:2*block_size) should be zeros
        assert (ref[:, :, block_size : 2 * block_size, :] == 0).all()
        # Other rows should be non-zero
        assert (ref[:, :, :block_size, :] != 0).any()
        assert (ref[:, :, 2 * block_size :, :] != 0).any()
