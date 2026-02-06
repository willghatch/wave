# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Fix data layout for chained MMAs where a 32x32 accumulator feeds into a
32x32x16 MMA input via permute.

The 32x32 accumulator layout has 4-consecutive element groups in terms of
M positions: [0,1,2,3], [8,9,10,11], [16,17,18,19], [24,25,26,27] for lanes 0-31
and [4,5,6,7], [12,13,14,15], [20,21,22,23], [28,29,30,31] for lanes 32-63.

After permute [K2, M] -> [M, K2], these M positions become K positions for the
second MMA. The 32x32x16 MMA requires 8 consecutive K values as input, but the
layout only provides 4 consecutive values at a time.

This pass marks permute nodes that need an inter-MMA shuffle. The shuffle is
applied during index propagation in index_sequence_analysis, which allows
downstream operations (like add, select for masks) to see the correct shuffled
indices and vector shapes.
"""

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    MMA,
    Permute,
    get_custom,
)
from .constraints import Constraint, HardwareConstraint, MMAType
from .utils.graph_utils import capture_backward_slice


# MMA types with 32x32 output that have 4-element consecutive groups
MMA_32x32_TYPES = {
    MMAType.F32_32x32x8_F16,
    MMAType.F32_32x32x16_F16,
    MMAType.F32_32x32x16_F8,
    MMAType.F32_32x32x16_BF16,
    MMAType.F32_32x32x16_K4_F8,
    MMAType.F32_32x32x16_K8_F16,
    MMAType.I32_32x32x8_I8,
    MMAType.I32_32x32x16_I8,
}

# MMA types that require 8 consecutive K elements as input
MMA_NEEDS_8_CONSECUTIVE = {
    MMAType.F32_32x32x16_F16,
    MMAType.F32_32x32x16_F8,
    MMAType.F32_32x32x16_BF16,
    MMAType.F32_32x32x16_K4_F8,
    MMAType.F32_32x32x16_K8_F16,
    MMAType.I32_32x32x16_I8,
}


def get_source_mma(node: fx.Node) -> MMA | None:
    """
    Find the source MMA by following data flow backward through any operations.
    Uses capture_backward_slice which handles IterArg, GetResult, Iterate,
    Conditional, and any chain of operations.
    Returns the first MMA found in the backward slice, or None if no MMA found.
    """
    # capture_backward_slice uses BFS via get_inputs() which properly handles
    # loop-carried variables, conditionals, and region boundaries
    backward_slice = list(capture_backward_slice(node))
    for arg in reversed(backward_slice):
        custom = get_custom(arg)
        if isinstance(custom, MMA):
            return custom
    return None


def find_permute_between_mmas(source_mma_node: fx.Node, target_mma: MMA) -> Permute | None:
    """
    Find a permute node in the data flow path from source MMA to target MMA.
    
    This searches forward from the source MMA through the backward slice of
    the target MMA's input. Returns the first Permute node found, or None.
    """
    # Get the input node to the target MMA that we're checking
    # This should be either lhs or rhs
    target_input = None
    if target_mma.lhs == source_mma_node or get_source_mma(target_mma.lhs) is not None:
        target_input = target_mma.lhs
    elif target_mma.rhs == source_mma_node or get_source_mma(target_mma.rhs) is not None:
        target_input = target_mma.rhs
    else:
        return None
    
    # Search backward from target input to find permute
    backward_slice = list(capture_backward_slice(target_input))
    for node in backward_slice:
        custom = get_custom(node)
        if isinstance(custom, Permute):
            return custom
    
    return None


def needs_shuffle_fix(
    source_mma: MMA,
    target_mma: MMA,
    hardware_constraint: HardwareConstraint,
) -> bool:
    """
    Check if the chained MMA pattern requires a shuffle fix.

    Returns True if:
    - Source MMA has 32x32 accumulator layout (4-consecutive groups)
    - Target MMA requires 8 consecutive K elements
    """
    source_type = source_mma.mma_type or hardware_constraint.mma_type
    target_type = target_mma.mma_type or hardware_constraint.mma_type

    return source_type in MMA_32x32_TYPES and target_type in MMA_NEEDS_8_CONSECUTIVE


def mark_permute_for_inter_mma_shuffle(
    permute: Permute, 
    source_mma: MMA,
    target_mma: MMA,
    hardware_constraint: HardwareConstraint
):
    """
    Mark a permute node as needing inter-MMA shuffle transformation.
    
    This metadata will be checked during index_sequence_analysis, where the
    permute's transform_index_forward will use the inter_mma_shuffle layout
    instead of the standard MMA output layout.
    
    Args:
        permute: The Permute node between two MMAs
        source_mma: The MMA producing the data
        target_mma: The MMA consuming the data
        hardware_constraint: Hardware configuration
    """
    permute.fx_node.meta["inter_mma_shuffle"] = {
        "source_mma_type": source_mma.mma_type or hardware_constraint.mma_type,
        "target_mma_type": target_mma.mma_type or hardware_constraint.mma_type,
        "threads_per_wave": hardware_constraint.threads_per_wave,
    }


def fix_chained_mma_permute(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    Fix data layout for chained MMAs where a 32x32 accumulator feeds into
    a 32x32x16 MMA input.

    This pass detects any data flow from one MMA to another (including through
    loop-carried variables, conditionals, and arbitrary operation chains) and
    marks permute nodes for inter-MMA shuffle when needed.

    The shuffle fix is required when:
    - Source MMA has 32x32 accumulator layout (4-consecutive element groups)
    - Target MMA requires 8 consecutive K elements as input
    - A permute operation exists between them

    The actual transformation is applied during index_sequence_analysis when
    propagating indices through the marked permute node.
    """
    # Get hardware constraint
    hardware_constraint = None
    for c in constraints:
        if isinstance(c, HardwareConstraint):
            hardware_constraint = c
            break

    if hardware_constraint is None:
        return

    # Find all MMA nodes
    mma_nodes = trace.walk(lambda node: isinstance(get_custom(node), MMA))
    if not mma_nodes:
        return

    # Check each MMA to see if its inputs come from another MMA through a permute
    for mma_node in mma_nodes:
        mma_custom = get_custom(mma_node)

        # Check LHS input
        source_mma = get_source_mma(mma_custom.lhs)
        if source_mma is not None:
            if needs_shuffle_fix(source_mma, mma_custom, hardware_constraint):
                # Find the permute node between them
                permute = find_permute_between_mmas(source_mma.fx_node, mma_custom)
                if permute is not None:
                    mark_permute_for_inter_mma_shuffle(
                        permute, source_mma, mma_custom, hardware_constraint
                    )

        # Check RHS input
        source_mma = get_source_mma(mma_custom.rhs)
        if source_mma is not None:
            if needs_shuffle_fix(source_mma, mma_custom, hardware_constraint):
                # Find the permute node between them
                permute = find_permute_between_mmas(source_mma.fx_node, mma_custom)
                if permute is not None:
                    mark_permute_for_inter_mma_shuffle(
                        permute, source_mma, mma_custom, hardware_constraint
                    )
