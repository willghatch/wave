# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from copy import deepcopy
from typing import Optional

import torch.fx as fx

from ..._support.indexing import IndexExpr, IndexSequence, IndexSymbol
from ..._support.tracing import CapturedTrace
from ...lang.global_symbols import *
from ...ops.wave_ops import (
    CustomOp,
    MMA,
    MMABase,
    Permute,
    Reshape,
    ScaledMMA,
    get_custom,
)
from ..constraints import (
    HardwareConstraint,
    MMAOperand,
    MMAType,
    ScaledMMAType,
)
from .graph_utils import capture_backward_slice
from .symbol_utils import subs_idxc
from .tag_utils import propagate_tag


def is_reshape_needed(
    node: CustomOp,
    node_vector_shapes: dict[IndexSymbol, int],
    vector_shapes: dict[IndexSymbol, int],
) -> bool:
    for dim in node.type.symbolic_shape:
        if dim not in vector_shapes:
            # Ignore nodes that are not used in both mmas.
            return False
        if node_vector_shapes[dim] != vector_shapes[dim]:
            return True
    return False


def get_mma_dimensional_mapping(
    trace: CapturedTrace,
    hardware_constraint: HardwareConstraint,
) -> tuple[
    dict[MMABase, dict[IndexSymbol, int]],
    dict[MMABase, dict[IndexSymbol, list[fx.Node]]],
]:
    """
    Given a trace, determine the MMA dimensional mapping for all the
    MMA operations in the graph. For example, if we have
        acc = tkw.mma(a_reg, b_reg, acc)
    where a_reg has shape UxV, b has shape SxV and acc has shape UxS,
    we map U to the MMA M dimension (0), S to the MMA N dimension (1) and
    V to the MMA K dimension (2). We maintain this map per mma node and
    also update the vector_shapes of the mma node based on this information.
    """

    def is_mma(node):
        return isinstance(get_custom(node), MMABase)

    mapping: dict[MMA, dict[IndexSymbol, int]] = {}

    mma_nodes = trace.walk(is_mma)
    for node in mma_nodes:
        custom: MMABase = get_custom(node)
        m, n = custom.acc_type.symbolic_shape[-2:]
        lhs_shape = custom.lhs_type.symbolic_shape
        rhs_shape = custom.rhs_type.symbolic_shape
        acc_shape = custom.acc_type.symbolic_shape

        try:
            reduction_dim_candidates = (set(lhs_shape) & set(rhs_shape)) - set(
                acc_shape
            )
            if len(reduction_dim_candidates) > 1:
                # Indicates we have batch dimensions as well.
                # Eliminate these using the vector shapes.
                for dim, value in hardware_constraint.vector_shapes.items():
                    if dim in reduction_dim_candidates and value == 0:
                        reduction_dim_candidates.remove(dim)
                assert (
                    len(reduction_dim_candidates) == 1
                ), f"Expected 1 reduction dimension, got {reduction_dim_candidates}"

            k = reduction_dim_candidates.pop()

        except KeyError as e:
            raise RuntimeError(
                f"{node}: Invalid MMA shapes\n{lhs_shape=}\n{rhs_shape=}\n{acc_shape=}\n{m=}, {n=}\n{custom}"
            )
        if m not in lhs_shape or n not in rhs_shape:
            raise RuntimeError(
                f"{node}: Invalid MMA shapes\n{lhs_shape=}\n{rhs_shape=}\n{acc_shape=}\n{m=}, {n=}, {k=}\n{custom}"
            )

        if isinstance(custom, ScaledMMA):
            lhs_scale_shape = custom.lhs_scale_type.symbolic_shape
            rhs_scale_shape = custom.rhs_scale_type.symbolic_shape
            try:
                scale_reduction_dim_candidates = (
                    set(lhs_scale_shape) & set(rhs_scale_shape)
                ) - set(acc_shape)
                if len(scale_reduction_dim_candidates) > 1:
                    # Indicates we have batch dimensions as well.
                    # Eliminate these using the vector shapes.
                    for dim, value in hardware_constraint.vector_shapes.items():
                        if dim in scale_reduction_dim_candidates and value == 0:
                            scale_reduction_dim_candidates.remove(dim)
                    assert (
                        len(scale_reduction_dim_candidates) == 1
                    ), f"Expected 1 reduction dimension, got {scale_reduction_dim_candidates}"

                k_scale = scale_reduction_dim_candidates.pop()
            except KeyError as e:
                raise RuntimeError(
                    f"{node}: Invalid Scaled MMA shapes\n{lhs_scale_shape=}\n{rhs_scale_shape=}\n{acc_shape=}\n{m=}, {n=}\n{custom}"
                )
            if m not in lhs_scale_shape or n not in rhs_scale_shape:
                raise RuntimeError(
                    f"{node}: Invalid Scaled MMA shapes\n{lhs_scale_shape=}\n{rhs_scale_shape=}\n{acc_shape=}\n{m=}, {n=}, {k=}\n{custom}"
                )

        if custom not in mapping:
            mapping[custom] = {}
        mapping[custom][m] = MMAOperand.M
        mapping[custom][n] = MMAOperand.N
        mapping[custom][k] = MMAOperand.K
        custom.vector_shapes = {
            m: hardware_constraint.mma_matrix_shapes(custom.mma_type)[0],
            n: hardware_constraint.mma_matrix_shapes(custom.mma_type)[1],
            k: hardware_constraint.mma_matrix_shapes(custom.mma_type)[2],
        }
        if isinstance(custom, ScaledMMA):
            mapping[custom][k_scale] = MMAOperand.K

        if hardware_constraint.vector_shapes:
            custom.vector_shapes.update(hardware_constraint.vector_shapes)
        custom.reduction_dim = k

        # Since expansion proceeds bottom-up, we set the vector shapes
        # of the parent reduction to the vector shapes of the last MMA node.
        if hasattr(custom.graph, "parent_op"):
            reduction = get_custom(custom.graph.parent_op)
            reduction.vector_shapes = custom.vector_shapes

    # Determine if any reshapes are required. Reshapes are added for
    # chained matmuls when the vector shapes of the operands in one matmul
    # differ from those in another matmul. The mma_slices contain all the ops
    # in the backward slice of the lhs and rhs upto a previous mma (if one exists).
    # So we check for the previous node of the first operator in the slice to see
    # if it is an MMA and if so check if a reshape is required.
    def add_reshape_if_needed(mma: MMABase, prev_mma: MMABase, arg_index: int):
        with mma.graph.inserting_before(mma.fx_node):
            arg = mma.lhs if arg_index == 0 else mma.rhs
            arg_custom = get_custom(arg)
            # Check if reshape is needed: compare prev_mma output with next mma input requirements
            if is_reshape_needed(arg_custom, prev_mma.vector_shapes, mma.vector_shapes):
                # Create reshape with target_vector_shape = prev_mma.vector_shapes
                target_vector_shape = deepcopy(prev_mma.vector_shapes)
                reshape_vector_shapes = deepcopy(mma.vector_shapes)
                has_inter_mma_shuffle = False
                
                # Check if there's inter_mma_shuffle in the backward slice
                # If so, adjust reshape_vector_shapes for the shuffled dimension
                # to enable proper slicing
                backward_slice = list(capture_backward_slice(arg))
                for node in backward_slice:
                    custom_node = get_custom(node)
                    if isinstance(custom_node, Permute):
                        inter_mma_meta = custom_node.fx_node.meta.get("inter_mma_shuffle", None)
                        if inter_mma_meta:
                            has_inter_mma_shuffle = True
                            # For inter-MMA shuffle (32x32x16 → 32x32x16):
                            # - The shuffled vector has 16 elements
                            # - It needs to be sliced into 8-element pieces for next MMA
                            # - num_partitions = target_vector_shape / reshape_vector_shapes
                            # - We want num_partitions = 2, so:
                            #   32 / reshape_vector_shapes[K2] = 2
                            #   reshape_vector_shapes[K2] = 16
                            #
                            # The dimension that needs slicing is the one that will be
                            # used as K input in the next MMA.
                            # For lhs (arg_index=0), K is the last dimension
                            # For rhs (arg_index=1), K is the second-to-last dimension
                            if arg_index == 0:
                                # LHS input: K is the last dimension
                                shuffle_dim = arg_custom.type.symbolic_shape[-1]
                            else:
                                # RHS input: K is the second-to-last dimension  
                                shuffle_dim = arg_custom.type.symbolic_shape[-2]
                            
                            # Only apply special slicing if target dimension is 32
                            # (for 32x32x16 MMA, not 32x32x8)
                            if (shuffle_dim in reshape_vector_shapes and 
                                shuffle_dim in target_vector_shape and
                                target_vector_shape[shuffle_dim] == 32 and
                                reshape_vector_shapes[shuffle_dim] == 32):
                                # Set reshape_vector_shapes to half to enable slicing
                                # num_partitions = 32 / 16 = 2 (slices 16-elem vector into two 8-elem pieces)
                                reshape_vector_shapes[shuffle_dim] = 16
                            break
                
                reshape = Reshape(arg, target_vector_shape).add_to_graph(
                    mma.graph, loc=mma.location
                )
                custom_reshape = get_custom(reshape)
                custom_reshape.vector_shapes = reshape_vector_shapes
                
                # Mark if this is an inter-MMA shuffle reshape
                if has_inter_mma_shuffle:
                    reshape.meta["inter_mma_shuffle_reshape"] = True
                
                propagate_tag(mma.fx_node, reshape)
                mma.update_arg(arg_index, reshape)

    def find_mma_in_slice(node: CustomOp) -> Optional[MMABase]:
        """
        Find the closest mma by iterating through the backward slice of a node
        in reverse.
        """
        slice = list(capture_backward_slice(node))
        for arg in reversed(slice):
            prev_mma = get_custom(arg)
            if isinstance(prev_mma, MMA):
                return prev_mma
        return None

    # Look in the backward slices of both the LHS and RHS to find
    # mmas. If found, add reshapes if necessary.
    for mma in mma_nodes:
        custom_mma = get_custom(mma)
        prev_mma = find_mma_in_slice(custom_mma.lhs)
        if prev_mma:
            add_reshape_if_needed(custom_mma, prev_mma, 0)
        prev_mma = find_mma_in_slice(custom_mma.rhs)
        if prev_mma:
            add_reshape_if_needed(custom_mma, prev_mma, 1)

    return mapping


def get_mfma_load_elems_per_thread(mfma_variant: MMAType | ScaledMMAType) -> int:
    match mfma_variant:
        case MMAType.RDNA4_WAVE32_F32_16x16x16_F16:
            return 8
        case MMAType.F32_16x16x16_F16 | MMAType.I32_16x16x16_I8:
            return 4
        case MMAType.F32_32x32x8_F16 | MMAType.I32_32x32x8_I8:
            return 4
        case (
            MMAType.F32_16x16x32_F8
            | MMAType.F32_16x16x32_BF16
            | MMAType.F32_16x16x32_F16
            | MMAType.F32_16x16x32_K8_F16
            | MMAType.F32_16x16x32_K4_F8
            | MMAType.I32_16x16x32_I8
        ):
            return 8
        case (
            MMAType.F32_32x32x16_F8
            | MMAType.F32_32x32x16_BF16
            | MMAType.F32_32x32x16_F16
            | MMAType.F32_32x32x16_K8_F16
            | MMAType.F32_32x32x16_K4_F8
            | MMAType.I32_32x32x16_I8
        ):
            return 8
        case ScaledMMAType.F32_16x16x128_F8F6F4:
            return 32
        case ScaledMMAType.F32_32x32x64_F8F6F4:
            return 32


def get_mfma_store_elems_per_thread(mfma_variant: MMAType | ScaledMMAType) -> int:
    match mfma_variant:
        case MMAType.RDNA4_WAVE32_F32_16x16x16_F16:
            return 8
        case MMAType.F32_16x16x16_F16 | MMAType.I32_16x16x16_I8:
            return 4
        case MMAType.F32_32x32x8_F16 | MMAType.I32_32x32x8_I8:
            return 16
        case (
            MMAType.F32_16x16x32_F8
            | MMAType.F32_16x16x32_BF16
            | MMAType.F32_16x16x32_F16
            | MMAType.F32_16x16x32_K8_F16
            | MMAType.F32_16x16x32_K4_F8
            | MMAType.I32_16x16x32_I8
        ):
            return 4
        case (
            MMAType.F32_32x32x16_F8
            | MMAType.F32_32x32x16_BF16
            | MMAType.F32_32x32x16_F16
            | MMAType.F32_32x32x16_K8_F16
            | MMAType.F32_32x32x16_K4_F8
            | MMAType.I32_32x32x16_I8
        ):
            return 16
        case ScaledMMAType.F32_16x16x128_F8F6F4:
            return 32
        case ScaledMMAType.F32_32x32x64_F8F6F4:
            return 32


def simplify_index(index: IndexExpr) -> IndexExpr:
    """
    Simplifies the index by applying the following bindings:
        - MMA acc_index bindings so the index of the MMA node is the acc_index.
    """
    mapping = {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 1}
    return subs_idxc(index.subs(mapping))


def specialize_index_sequence(
    index_seq: IndexSequence,
    mma_slices: dict[IndexSymbol, list[fx.Node]],
    custom: CustomOp,
) -> IndexSequence:
    """
    Given an index sequence, specialize it to a LHS, RHS or ACC index sequence
    based on whether the node is used as the LHS, RHS or ACC in the MMA node.
    If the node is not used as any of the operands, return the original index sequence
    with all the MMA symbols zeroed out.
    """
    if isinstance(custom, MMA):
        return index_seq
    operand_map = {MMA_LHS: 0, MMA_RHS: 0, MMA_ACC: 0}
    for key in mma_slices:
        if custom.fx_node in mma_slices[key]:
            operand_map[key] = 1
            return index_seq.subs(operand_map)
    return index_seq.subs(operand_map)
