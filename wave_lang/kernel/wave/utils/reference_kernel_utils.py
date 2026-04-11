# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention_bhsd(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    sliding_window: int = -1,
    custom_mask: Tensor | None = None,
) -> Tensor:
    """
    This version mimics PyTorch's `torch.nn.functional.scaled_dot_product_attention`
    with optional causal masking and improved numerical stability.
    Intended for comparison and debugging purposes.
    Args:
        query (Tensor): query tensor of shape [B, H, S_q, D].
        key (Tensor): key tensor of shape [B, H, S_k, D].
        value (Tensor): value tensor of shape [B, H, S_k, D].
        is_causal (bool): If True, applies causal masking to the attention logits.
    Returns:
        Tensor: Output tensor of shape [B, H, S_q, D] after applying attention.
    """
    if query.dtype != torch.float32:
        query = query.to(torch.float32)
    if key.dtype != torch.float32:
        key = key.to(torch.float32)
    if value.dtype != torch.float32:
        value = value.to(torch.float32)

    scale: float = query.shape[-1] ** -0.5
    attn_logits: Tensor = torch.matmul(query, key.transpose(-2, -1)) * scale

    if sliding_window >= 0:
        assert is_causal, f"Sliding window only supported with causal"

    if is_causal:
        seq_len_q, seq_len_k = attn_logits.shape[-2], attn_logits.shape[-1]
        causal_mask: Tensor = torch.tril(
            torch.ones(
                (seq_len_q, seq_len_k), device=attn_logits.device, dtype=torch.bool
            )
        )
        if sliding_window >= 0:
            causal_mask = causal_mask.triu(-sliding_window)
        attn_logits = attn_logits.masked_fill(~causal_mask, float("-inf"))

    if custom_mask is not None:
        bool_mask = custom_mask.to(torch.bool)
        bool_mask = bool_mask[:, None, :, None]
        assert bool_mask.shape == (query.shape[0], 1, query.shape[2], 1)
        attn_logits = attn_logits.masked_fill(bool_mask, float("-inf"))

    # Improve numerical stability using log-sum-exp trick
    attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values
    attn_weights: Tensor = F.softmax(attn_logits, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    return torch.matmul(attn_weights, value)


def sparse_scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_offsets: Tensor,
    block_indices: Tensor,
    block_size: int,
) -> Tensor:
    """Reference implementation of block-sparse attention.

    Computes dense attention but masks out blocks not in the sparse pattern.
    Used for correctness verification, not performance.

    Args:
        query: [B, H, S_q, D]
        key: [B, H, S_kv, D]
        value: [B, H, S_kv, D_v]
        block_offsets: [num_q_blocks + 1] CSR row pointers (int32)
        block_indices: [nnz_blocks] KV block indices (int32)
        block_size: block size in tokens
    Returns:
        Tensor: [B, H, S_q, D_v]
    """
    if query.dtype != torch.float32:
        query = query.to(torch.float32)
    if key.dtype != torch.float32:
        key = key.to(torch.float32)
    if value.dtype != torch.float32:
        value = value.to(torch.float32)

    S_q = query.shape[2]
    S_kv = key.shape[2]
    num_q_blocks = S_q // block_size
    num_kv_blocks = S_kv // block_size

    # Build element-level mask from block CSR pattern
    block_mask = torch.zeros(num_q_blocks, num_kv_blocks, dtype=torch.bool)
    for i in range(num_q_blocks):
        start = block_offsets[i].item()
        end = block_offsets[i + 1].item()
        if end > start:
            row_indices = block_indices[start:end].long()
            block_mask[i, row_indices] = True

    elem_mask = block_mask.repeat_interleave(block_size, dim=0).repeat_interleave(
        block_size, dim=1
    )
    # Broadcast to [B, H, S_q, S_kv]
    elem_mask = elem_mask.unsqueeze(0).unsqueeze(0).expand_as(
        torch.empty(query.shape[0], query.shape[1], S_q, S_kv)
    )
    elem_mask = elem_mask.to(query.device)

    scale: float = query.shape[-1] ** -0.5
    attn_logits = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_logits = attn_logits.masked_fill(~elem_mask, float("-inf"))

    # Numerical stability
    attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    return torch.matmul(attn_weights, value)
