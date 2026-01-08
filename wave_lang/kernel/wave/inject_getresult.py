# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
GetResult injection pass.

This pass ensures that all uses of Iterate and Conditional nodes are properly
wrapped with GetResult nodes. This establishes a key invariant that simplifies
downstream passes:

Invariant: Iterate/Conditional nodes should only have GetResult users,
except for metadata references (e.g., condition, start arguments).
"""

import torch.fx as fx
from wave_lang.support.logging import get_logger

from .._support.tracing import CapturedTrace
from ..ops.wave_ops import (
    Conditional,
    GetResult,
    Iterate,
    get_custom,
)

logger = get_logger("wave.inject_getresult")


def is_metadata_reference(user: fx.Node, iterate_node: fx.Node) -> bool:
    """
    Check if a user is a metadata-only reference to an Iterate/Conditional node.
    These include:
    - The 'start' or 'condition' arguments of Iterate/Conditional
    - References in implicit_captures lists (these are handled separately)
    """
    user_custom = get_custom(user)

    # Check if this is a reference to start/condition
    if isinstance(user_custom, (Iterate, Conditional)):
        if hasattr(user_custom, 'start') and user_custom.start == iterate_node:
            return True
        if hasattr(user_custom, 'condition') and user_custom.condition == iterate_node:
            return True

    return False


def determine_result_index(user: fx.Node, iterate_node: fx.Node) -> int:
    """
    Determine which result index is needed for a use of an Iterate/Conditional node.

    For now, we default to index 0. In the future, this could be made more sophisticated
    by analyzing the usage context.
    """
    iterate_custom = get_custom(iterate_node)

    # Check which argument position the iterate_node is in
    for i, arg in enumerate(user.args):
        if arg == iterate_node:
            # For now, assume the result index matches the argument position
            # or default to 0 if there's only one result
            if isinstance(iterate_custom.type, list) and i < len(iterate_custom.type):
                return i
            return 0

    # Check kwargs as well
    for key, arg in user.kwargs.items():
        if arg == iterate_node:
            return 0

    # Default to first result
    return 0


def inject_getresult_for_node(
    iterate_node: fx.Node,
    iterate_custom: Iterate | Conditional,
) -> int:
    """
    Inject GetResult nodes for all direct uses of an Iterate/Conditional node.

    Returns the number of GetResult nodes injected.
    """
    count = 0
    users = list(iterate_node.users.keys())

    for user in users:
        user_custom = get_custom(user)

        # Skip if already a GetResult
        if isinstance(user_custom, GetResult):
            continue

        # Skip metadata-only references
        if is_metadata_reference(user, iterate_node):
            continue

        # Determine which result index is needed
        result_idx = determine_result_index(user, iterate_node)

        # Inject GetResult node before the user
        with user.graph.inserting_before(user):
            logger.debug(
                f"Injecting GetResult for {iterate_node.name} (index {result_idx}) "
                f"before user {user.name}"
            )

            # Determine the type for this result
            if isinstance(iterate_custom.type, list):
                if result_idx < len(iterate_custom.type):
                    result_type = iterate_custom.type[result_idx]
                else:
                    result_type = iterate_custom.type[0]
            else:
                result_type = iterate_custom.type

            get_result = GetResult(iterate_node, result_idx).add_to_graph(
                user.graph,
                type=result_type,
                loc=iterate_custom.location if hasattr(iterate_custom, 'location') else None,
            )

            # Replace uses of iterate_node in this user with get_result
            user.replace_input_with(iterate_node, get_result)
            count += 1

    return count


def inject_missing_getresult_nodes(trace: CapturedTrace) -> int:
    """
    Walk all graphs in the trace and inject GetResult nodes for any direct uses
    of Iterate/Conditional nodes.

    Returns the total number of GetResult nodes injected.
    """
    total_count = 0

    # Process root graph
    root_graph = trace.get_root_graph()
    logger.debug(f"Processing root graph")

    for node in list(root_graph.nodes):
        custom = get_custom(node)
        if isinstance(custom, (Iterate, Conditional)):
            count = inject_getresult_for_node(node, custom)
            total_count += count
            if count > 0:
                logger.debug(
                    f"Injected {count} GetResult nodes for {node.name}"
                )

    # Process all subgraphs
    for subgraph_name, subgraph in trace.region_graph.subgraphs.items():
        logger.debug(f"Processing subgraph {subgraph_name}")

        for node in list(subgraph.nodes):
            custom = get_custom(node)
            if isinstance(custom, (Iterate, Conditional)):
                count = inject_getresult_for_node(node, custom)
                total_count += count
                if count > 0:
                    logger.debug(
                        f"Injected {count} GetResult nodes for {node.name} in {subgraph_name}"
                    )

    if total_count > 0:
        logger.info(f"Injected {total_count} GetResult nodes total")

    return total_count
