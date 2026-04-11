# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block-sparse attention kernel for Wave using CSR-style indirection.

Uses Approach A from the design doc: pad to max_blocks_per_row and mask.
The kv_token_indices tensor is 2D [query_seq_len, max_blocks_per_row *
BLOCK_K2], providing per-query-token indirection into the K/V tensors.
All Q blocks iterate the same number of times; padding is masked out.
"""

import math

import wave_lang.kernel.lang as tkl
import wave_lang.kernel.wave as tkw
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.constraints import MMAType

from .attention_common import AttentionShape


def prepare_sparse_attention_inputs(
    block_offsets,
    block_indices,
    num_q_blocks: int,
    block_size: int,
    query_seq_len: int,
):
    """Prepares the indirection tensors for the sparse attention kernel.

    Produces per-query-token indirection arrays by replicating block-level
    indices across all tokens within each Q block.

    Args:
        block_offsets: [num_q_blocks + 1] CSR row pointers
        block_indices: [nnz_blocks] KV block column indices
        num_q_blocks: number of Q blocks
        block_size: number of tokens per block (must equal BLOCK_K2)
        query_seq_len: total query sequence length

    Returns:
        (kv_token_indices, row_lengths, max_blocks_per_row)
        where:
        - kv_token_indices: [query_seq_len, max_blocks_per_row * block_size] int32
        - row_lengths: [query_seq_len] int32
          Number of valid KV tokens per query token (constant within a Q block).
        - max_blocks_per_row: int
    """
    import torch

    row_block_counts = block_offsets[1:] - block_offsets[:-1]
    max_blocks_per_row = int(row_block_counts.max().item()) if num_q_blocks > 0 else 0
    if max_blocks_per_row == 0:
        max_blocks_per_row = 1

    stride = max_blocks_per_row * block_size

    # Build per-Q-block indirection
    block_level_indices = torch.zeros(num_q_blocks, stride, dtype=torch.int32)
    block_level_lengths = torch.zeros(num_q_blocks, dtype=torch.int32)

    for q_blk in range(num_q_blocks):
        start = block_offsets[q_blk].item()
        end = block_offsets[q_blk + 1].item()
        block_level_lengths[q_blk] = (end - start) * block_size
        for local_i, global_i in enumerate(range(start, end)):
            kv_block = block_indices[global_i].item()
            t_start = local_i * block_size
            t_end = t_start + block_size
            block_level_indices[q_blk, t_start:t_end] = torch.arange(
                kv_block * block_size, (kv_block + 1) * block_size, dtype=torch.int32
            )

    # Replicate per-Q-block data to per-query-token
    kv_token_indices = block_level_indices.repeat_interleave(block_size, dim=0)
    row_lengths = block_level_lengths.repeat_interleave(block_size)

    # Ensure correct size
    if kv_token_indices.shape[0] < query_seq_len:
        pad_rows = query_seq_len - kv_token_indices.shape[0]
        kv_token_indices = torch.cat(
            [kv_token_indices, torch.zeros(pad_rows, stride, dtype=torch.int32)]
        )
        row_lengths = torch.cat(
            [row_lengths, torch.zeros(pad_rows, dtype=torch.int32)]
        )
    elif kv_token_indices.shape[0] > query_seq_len:
        kv_token_indices = kv_token_indices[:query_seq_len]
        row_lengths = row_lengths[:query_seq_len]

    return kv_token_indices, row_lengths, max_blocks_per_row


def get_sparse_bshd_attention_kernel(
    shape: AttentionShape,
    mfma_variant: list[MMAType],
    max_blocks_per_row: int,
    dynamic_dims: bool = False,
):
    """Creates a Wave kernel for block-sparse BSHD attention.

    Uses Approach A: iterate K2 over max_blocks_per_row * BLOCK_K2, using
    a 2D indirection tensor [M, K2] to read K/V from the correct positions.
    Padding iterations are masked with -inf bias.

    Input layout: BSHD (B=batch, S=seq, H=heads, D=head_dim)
      q: [B, S_q, H, D]  (B=1 for now)
      k: [B, S_kv, H, D]
      v: [B, S_kv, H, D_v]
      kv_indices: [S_q, max_blocks_per_row * BLOCK_K2]
      row_lens: [S_q]
      output: [B, S_q, H, D_v]

    Returns:
        (kernel_fn, hyperparams, dynamic_symbols)
    """
    B = tkl.sym.B
    M = tkl.sym.M
    N = tkl.sym.N
    K1 = tkl.sym.K1
    K2 = tkl.sym.K2     # max_blocks_per_row * BLOCK_K2
    H = tkl.sym.H
    N_KV = tkl.sym.N_KV
    BLOCK_B = tkl.sym.BLOCK_B
    BLOCK_M = tkl.sym.BLOCK_M
    BLOCK_N = tkl.sym.BLOCK_N
    BLOCK_K2 = tkl.sym.BLOCK_K2
    BLOCK_H = tkl.sym.BLOCK_H
    ADDRESS_SPACE = tkl.sym.ADDRESS_SPACE

    num_waves = 4
    block_k2 = 64

    constraints: list[tkw.Constraint] = [tkw.WorkgroupConstraint(M, BLOCK_M, 0)]
    constraints += [tkw.WorkgroupConstraint(N, BLOCK_N, 1)]
    constraints += [tkw.WorkgroupConstraint(B, BLOCK_B, 2)]
    constraints += [tkw.WorkgroupConstraint(H, BLOCK_H, 3)]
    constraints += [tkw.TilingConstraint(K2, BLOCK_K2)]
    constraints += [tkw.WaveConstraint(M, BLOCK_M / num_waves)]
    constraints += [tkw.WaveConstraint(N, BLOCK_N / 1)]

    if mfma_variant[1] == MMAType.F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
        TPW = 64
    if mfma_variant[1] == MMAType.F32_32x32x8_F16:
        Mvec = 32
        Nvec = 32
        TPW = 64
    if mfma_variant[1] == MMAType.RDNA4_WAVE32_F32_16x16x16_F16:
        Mvec = 16
        Nvec = 16
        TPW = 32

    constraints += [
        tkw.HardwareConstraint(
            threads_per_wave=TPW,
            mma_type=mfma_variant[1],
            vector_shapes={B: 0, H: 0, M: Mvec, N: Nvec},
        )
    ]

    if dynamic_dims:
        constraints += [tkw.Assumption(K2 > BLOCK_K2 * 4)]

    i = tkw.IndexMapping.iterator(0)
    j = tkw.IndexMapping.iterator(1)
    k = tkw.IndexMapping.iterator(2)
    l = tkw.IndexMapping.iterator(3)
    d0 = tkw.IndexMapping.dynamic_val(0)

    # Output: register [B, H, N, M] -> memory [B, M, H, N]
    output_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, N: k, M: l},
        outputs={B: i, M: l, H: j, N: k},
    )

    # Q: [B, M, H, K1]
    q_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, H: j, M: k, K1: l},
        outputs={B: i, H: j, M: k, K1: l},
    )

    # K with indirection: reads from [B, N_KV, H, K1]
    sparse_k_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, N_KV: d0, H: j, K1: l},
        outputs={B: i, H: j, K2: k, K1: l},
        dynamic_val_mappings={K2: k},
    )

    # V with indirection: reads from [B, N_KV, H, N]
    sparse_v_mapping = tkw.IndexMapping(
        num_iterators=4,
        inputs={B: i, N_KV: d0, H: j, N: k},
        outputs={B: i, H: j, N: k, K2: l},
        dynamic_val_mappings={K2: l},
    )

    log2e = 1.44269504089
    dk_sqrt = math.sqrt(1.0 / shape.head_size)

    @tkw.wave(constraints)
    def sparse_attention(
        q: tkl.Memory[B, M, H, K1, GLOBAL_ADDRESS_SPACE, tkl.f16],
        k_mem: tkl.Memory[B, N_KV, H, K1, ADDRESS_SPACE, tkl.f16],
        v_mem: tkl.Memory[B, N_KV, H, N, ADDRESS_SPACE, tkl.f16],
        kv_indices: tkl.Memory[M, K2, GLOBAL_ADDRESS_SPACE, tkl.i32],
        row_lens: tkl.Memory[M, GLOBAL_ADDRESS_SPACE, tkl.i32],
        c: tkl.Memory[B, M, H, N, GLOBAL_ADDRESS_SPACE, tkl.f32],
    ):
        qkv_scaling = tkl.Register[B, H, M, K1, tkl.f16](dk_sqrt * log2e)
        c_reg = tkl.Register[B, H, N, M, tkl.f32](0.0)
        init_sum = tkl.Register[B, H, M, tkl.f32](0.0)
        init_max = tkl.Register[B, H, M, tkl.f32](-1e6)
        ZEROF = tkl.Register[M, K2, tkl.f32](0.0)
        MIN_INF = tkl.Register[M, K2, tkl.f32](-1e6)

        @tkw.iterate(K2, init_args=[init_max, init_sum, c_reg])
        def repeat(
            partial_max: tkl.Register[B, H, M, tkl.f32],
            partial_sum: tkl.Register[B, H, M, tkl.f32],
            acc: tkl.Register[B, H, N, M, tkl.f32],
        ):
            imm_reg = tkl.Register[B, H, K2, M, tkl.f32](0.0)
            q_reg = tkw.read(q, mapping=q_mapping)
            q_reg *= qkv_scaling

            # Read indirection indices for this tile
            kv_idx_k = tkw.read(kv_indices)
            kv_idx_v = tkw.read(kv_indices)

            # Read K/V using indirection
            k_reg = tkw.read(
                k_mem,
                mapping=sparse_k_mapping,
                mapping_dynamic_vals=(kv_idx_k,),
            )
            v_reg = tkw.read(
                v_mem,
                mapping=sparse_v_mapping,
                mapping_dynamic_vals=(kv_idx_v,),
            )

            inner_acc = tkw.mma(k_reg, q_reg, imm_reg, mfma_variant[0])
            x_j = tkw.permute(inner_acc, target_shape=[B, H, M, K2])

            # Mask out padding iterations
            k2_index = tkw.self_index(K2, tkl.i32)
            valid_len = tkw.read(row_lens)
            mask = k2_index < valid_len
            mask = tkw.broadcast(mask, target_shape=[M, K2])
            mask = tkw.cast(mask, tkw.i1)
            bias = tkw.select(mask, ZEROF, MIN_INF)
            x_j = x_j + bias

            m_j = tkw.max(x_j, partial_max, dim=K2)
            e_delta_max = tkw.exp2(partial_max - m_j)
            e_delta = tkw.exp2(x_j - m_j)
            e_init = partial_sum * e_delta_max
            d_j = tkw.sum(e_delta, e_init, dim=K2)
            imm_f16 = tkw.cast(e_delta, tkl.f16)
            new_acc = acc * e_delta_max
            acc = tkw.mma(v_reg, imm_f16, new_acc)
            return m_j, d_j, acc

        res_max, res_sum, res_mm = repeat
        reciprocal_sum = tkw.reciprocal(res_sum)

        # NaN guard for fully-masked Q blocks
        is_nan = res_sum == init_sum
        is_nan = tkw.cast(is_nan, tkw.i1)
        reciprocal_sum = tkw.select(is_nan, init_sum, reciprocal_sum)

        res = res_mm * reciprocal_sum
        tkw.write(res, c, mapping=output_mapping)

    k2_size = max_blocks_per_row * block_k2

    hyperparams = {
        ADDRESS_SPACE: SHARED_ADDRESS_SPACE,
        BLOCK_B: 1,
        BLOCK_H: 1,
        BLOCK_M: 128,
        BLOCK_N: 64,
        BLOCK_K2: block_k2,
        B: 1,
        H: shape.num_query_heads,
        M: shape.query_seq_len,
        N: shape.head_size_kv,
        K1: shape.head_size,
        K2: k2_size,
        N_KV: shape.kv_seq_len,
    }

    dynamic_symbols = []
    if dynamic_dims:
        dynamic_symbols.append(M)
        dynamic_symbols.append(N)
        dynamic_symbols.append(B)
        dynamic_symbols.append(K2)
        del hyperparams[M]
        del hyperparams[N]
        del hyperparams[B]
        del hyperparams[K2]

    return sparse_attention, hyperparams, dynamic_symbols
