# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Flatten N-D read indices to 1-D physical offsets (LINEAR_INDEX).

For every eligible Read (unmapped and mapped), this pass:

1. Resolves the index mapping (if any) into physical coordinates.
2. Linearizes them into a single flat offset using memory strides.
3. Converts bounds to expression-keyed form via ``delinearize_index``.
4. Replaces the index with ``{LINEAR_INDEX: IndexSequence(flat, ept, 1)}``.

Reads with shared-memory targets are skipped.  Reads with
``mapping_dynamic_vals`` are flattened normally; the dynamic val symbols
(``$dynamic_val0``, etc.) remain as free symbols in the flat expression.

For mapped reads the flat expression is intentionally left unsimplified
so that ``gen_sympy_index`` lowers each floor/Mod term independently,
producing correct integer MLIR ops.  Algebraic simplification (e.g.
``mem_simplify``) is only applied to unmapped reads where the
expressions are simple enough not to cause floor/Mod mismatches.

IV stride extraction is NOT done here -- that happens in the separate
``annotate_iv_strides`` post-merge pass.
"""

from collections.abc import Sequence

import sympy

from ..._support.indexing import IndexingContext, IndexSequence
from ..._support.tracing import CapturedTrace
from ...compiler.utils import strides_from_symbolic_shape
from ...lang.global_symbols import LINEAR_INDEX, SHARED_ADDRESS_SPACE
from ...ops.wave_ops import Read, get_custom
from ..assumptions import get_divisibility_subs
from ..constraints import Constraint
from ..utils.general_utils import (
    delinearize_index,
    infer_dim,
    is_flattened_index,
)
from ..utils.mapping_utils import (
    _infer_floor_to_exact,
    linearize_dims,
    mem_simplify,
    transform_index_on_mapping,
)
from ..utils.symbol_utils import subs_idxc


def _convert_bounds(bounds, flat_start, ept, symbolic_shape, symbolic_dims):
    """Convert per-dim bounds to expression-keyed form.

    Delinearizes ``flat_start + iota(ept)`` back to per-dim coordinates
    and maps each bounded dim to its delinearized expression.
    Returns ``None`` if there are no applicable bounds.
    """
    idxc = IndexingContext.current()
    flat_with_iota = flat_start + idxc.iota(ept)
    shape_sizes = [subs_idxc(d) for d in symbolic_shape]
    coords = delinearize_index(flat_with_iota, shape_sizes)

    new_bounds = {}
    for dim, bound in bounds.items():
        if dim not in symbolic_dims:
            continue
        dim_idx = symbolic_dims.index(dim)
        new_bounds[coords[dim_idx]] = bound
    return new_bounds or None


def _get_physical_starts(custom, symbolic_shape, symbolic_dims):
    """Return per-dim physical start expressions for a Read.

    For mapped reads, applies the mapping via ``transform_index_on_mapping``.
    For unmapped / identity-mapped reads, reads starts directly from the index.
    Returns ``None`` when required dimensions are missing.
    """
    if custom.mapping is not None and not custom.has_identity_mapping():
        transformed = transform_index_on_mapping(
            custom.mapping, symbolic_shape, custom.index, is_read=True
        )
        if not all(dim in transformed for dim in symbolic_dims):
            return None
        return {dim: transformed[dim] for dim in symbolic_dims}
    if not all(dim in custom.index for dim in symbolic_dims):
        return None
    return {
        dim: (
            custom.index[dim].start
            if isinstance(custom.index[dim], IndexSequence)
            else custom.index[dim]
        )
        for dim in symbolic_dims
    }


def flatten_read_indices(
    trace: CapturedTrace,
    constraints: Sequence[Constraint] = (),
):
    """Flatten N-D read indices to 1-D LINEAR_INDEX for all eligible Reads."""
    idxc = IndexingContext.current()
    div_fwd, div_bwd = get_divisibility_subs(constraints)

    for node in trace.walk(lambda n: isinstance(get_custom(n), Read)):
        custom = get_custom(node)

        index = custom.index
        mem_node = custom.memory
        bounds = custom.bounds

        if is_flattened_index(index):
            continue

        memory = get_custom(mem_node)
        if (
            hasattr(memory, "type")
            and hasattr(memory.type, "address_space")
            and subs_idxc(memory.type.address_space) == SHARED_ADDRESS_SPACE
        ):
            continue

        symbolic_shape = memory.type.symbolic_shape
        symbolic_dims = [infer_dim(d) for d in symbolic_shape]

        has_mapping = custom.mapping is not None and not (
            hasattr(custom, "has_identity_mapping")
            and custom.has_identity_mapping()
        )

        phys_starts = _get_physical_starts(custom, symbolic_shape, symbolic_dims)
        if phys_starts is None:
            continue

        mem_strides = list(
            strides_from_symbolic_shape(
                idxc, symbolic_shape, allow_mixed_shapes=True
            )
        )

        if has_mapping:
            # Mapped reads: build the flat expression WITHOUT algebraic
            # simplification.  The raw sum-of-products preserves per-dim
            # floor/Mod structure so gen_sympy_index lowers each term to
            # the correct integer MLIR op.
            dim_exprs = [sympy.sympify(phys_starts[dim]) for dim in symbolic_dims]
            flat_start = sum(
                expr * stride
                for expr, stride in zip(dim_exprs, mem_strides)
            )
        else:
            dim_exprs = [sympy.sympify(phys_starts[dim]) for dim in symbolic_dims]
            dim_exprs = [subs_idxc(e) for e in dim_exprs]

            if div_fwd:
                fwd_dict = dict(div_fwd)
                dim_exprs = [sympy.sympify(e).subs(fwd_dict) for e in dim_exprs]
                applied_strides = [sympy.sympify(s).subs(fwd_dict) for s in mem_strides]
            else:
                floor_subs = _infer_floor_to_exact(mem_strides)
                if floor_subs:
                    dim_exprs = [sympy.sympify(e).subs(floor_subs) for e in dim_exprs]
                applied_strides = mem_strides

            flat_start = linearize_dims(dim_exprs, applied_strides)

            if div_bwd:
                bwd_dict = dict(div_bwd)
                flat_start = mem_simplify(sympy.sympify(flat_start).subs(bwd_dict))

        ept = custom.elements_per_thread
        ept_val = subs_idxc(ept) if not isinstance(ept, int) else ept

        new_bounds = None
        if bounds:
            new_bounds = _convert_bounds(
                bounds, flat_start, ept_val, symbolic_shape, symbolic_dims,
            )

        new_index = {LINEAR_INDEX: IndexSequence(flat_start, ept_val, 1)}

        custom.index = new_index
        custom.update_arg("mapping", None)
        custom.update_arg("bounds", new_bounds)
