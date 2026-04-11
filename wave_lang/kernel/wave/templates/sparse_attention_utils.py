# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Utilities for generating CSR block-sparse patterns for sparse attention.

These functions convert high-level sparsity descriptions (dense, causal,
local window, etc.) into CSR-format (block_offsets, block_indices) arrays
suitable for the sparse attention kernel.
"""

import torch


def dense_block_pattern(
    num_q_blocks: int,
    num_kv_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (block_offsets, block_indices) for full attention.

    Every Q block attends to every KV block.
    """
    offsets = torch.arange(
        0, (num_q_blocks + 1) * num_kv_blocks, num_kv_blocks, dtype=torch.int32
    )
    indices = torch.arange(num_kv_blocks, dtype=torch.int32).repeat(num_q_blocks)
    return offsets, indices


def causal_block_pattern(
    num_q_blocks: int,
    num_kv_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (block_offsets, block_indices) for causal attention.

    Q block i attends to KV blocks 0..min(i, num_kv_blocks-1).
    """
    all_indices = []
    counts = []
    for i in range(num_q_blocks):
        end = min(i, num_kv_blocks - 1)
        row_indices = torch.arange(end + 1, dtype=torch.int32)
        all_indices.append(row_indices)
        counts.append(len(row_indices))

    offsets = torch.zeros(num_q_blocks + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(torch.tensor(counts, dtype=torch.int32), dim=0)
    indices = (
        torch.cat(all_indices) if all_indices else torch.tensor([], dtype=torch.int32)
    )
    return offsets, indices


def local_window_block_pattern(
    num_q_blocks: int,
    num_kv_blocks: int,
    window_blocks: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (block_offsets, block_indices) for local window attention.

    If causal, Q block i attends to max(0, i-w+1)..i.
    If not causal, Q block i attends to max(0, i-w//2)..min(num_kv-1, i+w//2).
    """
    all_indices = []
    counts = []
    for i in range(num_q_blocks):
        if causal:
            start = max(0, i - window_blocks + 1)
            end = i
        else:
            half = window_blocks // 2
            start = max(0, i - half)
            end = min(num_kv_blocks - 1, i + half)
        row_indices = torch.arange(start, end + 1, dtype=torch.int32)
        all_indices.append(row_indices)
        counts.append(len(row_indices))

    offsets = torch.zeros(num_q_blocks + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(torch.tensor(counts, dtype=torch.int32), dim=0)
    indices = (
        torch.cat(all_indices) if all_indices else torch.tensor([], dtype=torch.int32)
    )
    return offsets, indices


def block_pattern_with_global_tokens(
    base_offsets: torch.Tensor,
    base_indices: torch.Tensor,
    num_q_blocks: int,
    num_kv_blocks: int,
    num_global_kv_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adds the first num_global_kv_blocks KV blocks to every Q block's
    attention set (union with existing pattern).  Deduplicates and sorts."""
    base_mask = block_sparse_to_dense_mask(
        base_offsets, base_indices, num_q_blocks, num_kv_blocks
    )
    base_mask[:, :num_global_kv_blocks] = True
    return dense_mask_to_block_sparse(base_mask)


def dense_mask_to_block_sparse(
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts a dense block-level mask to CSR format.

    mask[i][j] == True means Q block i attends to KV block j.
    """
    num_q_blocks = mask.shape[0]
    all_indices = []
    counts = []
    for i in range(num_q_blocks):
        row_indices = torch.where(mask[i])[0].to(torch.int32)
        all_indices.append(row_indices)
        counts.append(len(row_indices))

    offsets = torch.zeros(num_q_blocks + 1, dtype=torch.int32)
    offsets[1:] = torch.cumsum(torch.tensor(counts, dtype=torch.int32), dim=0)
    if all_indices and sum(counts) > 0:
        indices = torch.cat(all_indices)
    else:
        indices = torch.tensor([], dtype=torch.int32)
    return offsets, indices


def block_sparse_to_dense_mask(
    block_offsets: torch.Tensor,
    block_indices: torch.Tensor,
    num_q_blocks: int,
    num_kv_blocks: int,
) -> torch.Tensor:
    """Inverse of dense_mask_to_block_sparse.  For testing round-trips."""
    mask = torch.zeros(num_q_blocks, num_kv_blocks, dtype=torch.bool)
    for i in range(num_q_blocks):
        start = block_offsets[i].item()
        end = block_offsets[i + 1].item()
        if end > start:
            row_indices = block_indices[start:end].long()
            mask[i, row_indices] = True
    return mask
