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

This pass marks nodes that need a shuffle fix, which is then handled during
code generation in the Reshape handler.
"""

import torch.fx as fx

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    MMA,
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


def mark_node_for_shuffle_fix(node: fx.Node, hardware_constraint: HardwareConstraint):
    """
    Mark a node as needing the chained MMA shuffle fix.
    The actual fix is applied during code generation.
    
    Also updates the index to reflect the shuffled layout:
    - Before shuffle: 4-element groups with expression like Mod($GPR_NUM, 4) + 8*(Mod(floor($GPR_NUM/4), 4))
    - After shuffle: 8-element groups with expression like 8*floor(Mod($T0, 64)/32) + Mod($GPR_NUM, 8)
    
    Note: The full expression also includes 16*floor($GPR_NUM/8) for the second 8-element group,
    but that's handled during expansion when we know which partition is being extracted.
    """
    node.meta["chained_mma_shuffle_fix"] = {
        "threads_per_wave": hardware_constraint.threads_per_wave,
    }
    
    # Update the index of this node and its users (like reshape) to reflect the shuffled layout
    # This needs to happen after indices are set but before expansion
    # We'll do this in a separate pass that runs after set_node_indices
    node.meta["shuffle_fix_index_update_needed"] = True


def fix_chained_mma_permute(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    Fix data layout for chained MMAs where a 32x32 accumulator feeds into
    a 32x32x16 MMA input.

    This pass detects any data flow from one MMA to another (including through
    loop-carried variables, conditionals, and arbitrary operation chains) and
    marks the input node for shuffle fix when needed.

    The shuffle fix is required when:
    - Source MMA has 32x32 accumulator layout (4-consecutive element groups)
    - Target MMA requires 8 consecutive K elements as input

    The actual fix is applied during code generation in the Reshape handler.
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

    # Check each MMA to see if its inputs come from a problematic permute chain
    for mma_node in mma_nodes:
        mma_custom = get_custom(mma_node)

        # Check LHS input
        source_mma = get_source_mma(mma_custom.lhs)
        if source_mma is not None:
            if needs_shuffle_fix(source_mma, mma_custom, hardware_constraint):
                mark_node_for_shuffle_fix(mma_custom.lhs, hardware_constraint)

        # Check RHS input
        source_mma = get_source_mma(mma_custom.rhs)
        if source_mma is not None:
            if needs_shuffle_fix(source_mma, mma_custom, hardware_constraint):
                mark_node_for_shuffle_fix(mma_custom.rhs, hardware_constraint)
