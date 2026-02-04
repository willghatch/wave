# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Update indices for nodes that have been marked for shuffle fix.

After the chained MMA shuffle fix is applied, the data layout changes from
4-element groups to 8-element groups. The index expressions need to be updated
to reflect this new layout.

This pass runs after set_node_indices and fix_chained_mma_permute, but before
expand_graph.
"""

import sympy
import torch.fx as fx

from .._support.indexing import IndexSymbol, IndexSequence
from .._support.tracing import CapturedTrace
from ..ops.wave_ops import get_custom, Reshape
from .constraints import Constraint

import logging

logger = logging.getLogger(__name__)


def update_shuffled_indices(
    trace: CapturedTrace,
    constraints: list[Constraint],
):
    """
    Update indices for nodes marked with shuffle_fix_index_update_needed.
    
    The shuffle fix changes the data layout from 4-element groups to 8-element groups.
    The index expression needs to be updated from:
      Mod($GPR_NUM, 4) + 8*(Mod(floor($GPR_NUM/4), 4)) + 4*floor((Mod($T0, 64))/32)
    to:
      8*floor(Mod($T0, 64)/32) + Mod($GPR_NUM, 8) + 16*floor($GPR_NUM/8)
    
    However, since expansion hasn't happened yet, we don't know which 8-element
    partition will be extracted. So we update to a base expression and let
    expansion add the partition offset.
    """
    
    logger.debug("Running update_shuffled_indices pass")
    
    T0 = IndexSymbol("T0")
    GPR_NUM = IndexSymbol("GPR_NUM")
    
    def update_node_index(node: fx.Node):
        if not node.meta.get("shuffle_fix_index_update_needed"):
            return False
        
        custom = get_custom(node)
        if not hasattr(custom, 'index') or custom.index is None:
            logger.debug(f"Node {node.name} marked for shuffle fix but has no index")
            return False
        
        # Find the innermost dimension with size > 1 (this is the K dimension)
        innermost_dim = None
        for dim in reversed(custom.type.symbolic_shape):
            if dim in custom.index and custom.index[dim].size > 1:
                innermost_dim = dim
                break
        
        if innermost_dim is None:
            logger.debug(f"Node {node.name} has no innermost dimension with size > 1")
            return False
        
        old_index = custom.index[innermost_dim]
        old_start = old_index.start
        
        # Extract the block offset (terms that don't depend on T0 or GPR_NUM)
        # We need to be careful to exclude ALL thread-dependent terms
        if isinstance(old_start, sympy.Expr):
            block_offset_terms = []
            for term in sympy.Add.make_args(old_start):
                # Check if this term contains T0 or GPR_NUM (case-insensitive)
                term_symbols = {str(s) for s in term.free_symbols}
                if not any(s in term_symbols for s in ['T0', 'GPR_NUM', '$T0', '$GPR_NUM']):
                    block_offset_terms.append(term)
            block_offset = sum(block_offset_terms) if block_offset_terms else 0
        else:
            block_offset = old_start if isinstance(old_start, int) else 0
        
        # New thread-dependent offset for 8-element grouped layout
        # This is the base expression; expansion will add 16*floor($GPR_NUM/8) for the second partition
        thread_offset = 8 * sympy.floor(sympy.Mod(T0, 64) / 32) + sympy.Mod(GPR_NUM, 8)
        new_start = block_offset + thread_offset
        
        # Update the index: size is now 8 (one 8-element group), stride is 1 (consecutive)
        custom.index[innermost_dim] = IndexSequence(
            new_start,
            8,  # Size is 8 (one 8-element group)
            1   # Stride is 1 (consecutive elements after shuffle)
        )
        
        logger.debug(f"Updated index for shuffled node {node.name}: {custom.index[innermost_dim]}")
        
        # Don't update the input node - the shuffle happens in the reshape,
        # so only the reshape output should have the new layout
        
        return False
    
    trace.walk(update_node_index)
