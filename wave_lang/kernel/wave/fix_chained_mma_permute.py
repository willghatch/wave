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
    CastOp,
    MMA,
    Permute,
    Reshape,
    get_custom,
)
from .constraints import Constraint, HardwareConstraint, MMAType


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
    Trace back through Reshape, Permute and Cast operations to find the source MMA.
    Returns None if the chain doesn't match the expected pattern.
    """
    custom = get_custom(node)

    # Handle Reshape -> Cast -> Permute -> MMA chain (after expansion)
    if isinstance(custom, Reshape):
        # Reshape.args is a list
        args = custom.args
        if isinstance(args, (list, tuple)) and len(args) > 0:
            return get_source_mma(args[0])
        return None
    elif isinstance(custom, CastOp):
        return get_source_mma(custom.arg)
    elif isinstance(custom, Permute):
        arg_custom = get_custom(custom.arg)
        if isinstance(arg_custom, MMA):
            return arg_custom
        # Could be MMA -> Cast -> Permute
        elif isinstance(arg_custom, CastOp):
            cast_arg = get_custom(arg_custom.arg)
            if isinstance(cast_arg, MMA):
                return cast_arg
        # Could also be MMA -> Permute (no cast)
        return get_source_mma(custom.arg)
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
    """
    node.meta["chained_mma_shuffle_fix"] = {
        "threads_per_wave": hardware_constraint.threads_per_wave,
    }


def fix_chained_mma_permute(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    Fix data layout for chained MMAs where a 32x32 accumulator feeds into
    a 32x32x16 MMA input via permute.

    This pass detects the pattern:
        MMA(32x32) -> Permute -> [Cast] -> MMA(32x32x16)

    And marks the permute/cast node so that the Reshape handler in codegen
    can emit the necessary shuffle operations.
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
