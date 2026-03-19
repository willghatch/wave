# Copyright 2025 The IREE Authors
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Flatten N-D read indices to 1-D physical offsets (LINEAR_INDEX).

Converts Read ops from an N-D logical index to a single linearised physical
offset, extracting the IV stride and packaging everything into
``{LINEAR_INDEX: IndexSequence(base_offset, ept, iv_stride)}``.

Handles both unmapped reads (identity or no mapping) and mapped reads
(non-identity mappings without dynamic vals).

For unmapped reads, the pass produces a monolithic flat offset that
codegen uses directly.

For mapped reads (preshuffle etc.), the pass applies the mapping to get
physical coordinates, extracts the IV stride via numerical probing, and
stores the stride in ``meta["iv_stride"]`` while keeping the per-dimension
physical index.  Codegen linearises per-dimension at the MLIR level to
avoid floor/Mod simplification mismatches between SymPy and MLIR integer
arithmetic.

This pass replaces ``annotate_iv_strides``.
"""

from collections.abc import Sequence

import sympy

from ..._support.indexing import IndexingContext, IndexSequence
from ..._support.tracing import CapturedTrace
from ...compiler.utils import strides_from_symbolic_shape
from ...lang.global_symbols import LINEAR_INDEX, SHARED_ADDRESS_SPACE
from ...ops.wave_ops import Read, GatherToLDS, get_custom
from ..assumptions import get_divisibility_subs
from ..constraints import Constraint
from ..utils.general_utils import (
    delinearize_index,
    infer_dim,
    is_flattened_index,
)
from ..utils.mapping_utils import (
    _infer_floor_to_exact,
    compute_iv_stride_through_mapping,
    linearize_dims,
    mem_simplify,
    transform_index_on_mapping,
)
from ..utils.symbol_utils import subs_idxc

_INDUCTION_PREFIX = "$ARG"


def _get_iv_symbols(flat_addr: sympy.Expr) -> list[sympy.Symbol]:
    """Return all induction-variable symbols in *flat_addr*."""
    return [s for s in flat_addr.free_symbols if str(s).startswith(_INDUCTION_PREFIX)]


def _extract_iv_stride(
    flat_addr: sympy.Expr,
    iv_sym: sympy.Symbol,
) -> tuple[sympy.Expr, sympy.Expr] | None:
    """Extract base_offset and iv_stride from flat_addr = base + iv * stride.

    Returns (base_offset, iv_stride) or None if the address is not affine
    in iv_sym.
    """
    diff = sympy.expand(flat_addr.subs(iv_sym, iv_sym + 1) - flat_addr)
    diff = mem_simplify(diff)
    if diff.free_symbols - {iv_sym}:
        if diff.is_Integer or diff.is_Number:
            pass
        else:
            return None
    if isinstance(diff, (int, sympy.Integer)):
        base_offset = flat_addr.subs(iv_sym, 0)
        return mem_simplify(base_offset), diff
    return None


def _flatten_unmapped_read(custom, index, mem_node, bounds, idxc, div_fwd, div_bwd):
    """Flatten an unmapped (identity or no mapping) Read to LINEAR_INDEX.

    Returns True if flattened successfully.
    """
    memory = get_custom(mem_node)
    mem_sym_shape = memory.type.symbolic_shape
    symbolic_dims = [infer_dim(d) for d in mem_sym_shape]

    mem_strides = list(strides_from_symbolic_shape(
        idxc, mem_sym_shape, allow_mixed_shapes=True
    ))

    phys_starts = []
    for dim in symbolic_dims:
        seq = index.get(dim)
        if seq is None:
            phys_starts.append(0)
        elif isinstance(seq, IndexSequence):
            phys_starts.append(seq.start)
        else:
            phys_starts.append(seq)

    dim_exprs = [sympy.sympify(s) for s in phys_starts]
    dim_exprs = [subs_idxc(e) for e in dim_exprs]

    if div_fwd:
        fwd_dict = dict(div_fwd)
        dim_exprs = [sympy.sympify(e).subs(fwd_dict) for e in dim_exprs]
        applied_mem_strides = [sympy.sympify(s).subs(fwd_dict) for s in mem_strides]
    else:
        floor_subs = _infer_floor_to_exact(mem_strides)
        if floor_subs:
            dim_exprs = [sympy.sympify(e).subs(floor_subs) for e in dim_exprs]
        applied_mem_strides = mem_strides

    flat_addr = linearize_dims(dim_exprs, applied_mem_strides)

    if div_bwd:
        bwd_dict = dict(div_bwd)
        flat_addr = mem_simplify(sympy.sympify(flat_addr).subs(bwd_dict))

    ept = custom.elements_per_thread
    ept_val = subs_idxc(ept) if not isinstance(ept, int) else ept

    iv_syms = _get_iv_symbols(flat_addr)
    base_offset = flat_addr
    iv_stride = 0

    if iv_syms:
        iv_sym = iv_syms[0]
        result = _extract_iv_stride(flat_addr, iv_sym)
        if result is not None:
            base_offset, iv_stride = result
        else:
            return False

    new_bounds = None
    if bounds:
        flat_with_iota = flat_addr + idxc.iota(ept_val)
        shape_sizes = [subs_idxc(d) for d in mem_sym_shape]
        coords = delinearize_index(flat_with_iota, shape_sizes)
        new_bounds = {}
        for dim, bound in bounds.items():
            if dim not in symbolic_dims:
                continue
            dim_idx = symbolic_dims.index(dim)
            new_bounds[coords[dim_idx]] = bound

    new_index = {LINEAR_INDEX: IndexSequence(base_offset, ept_val, iv_stride)}
    custom.index = new_index
    custom.update_arg("mapping", None)
    custom.update_arg("bounds", new_bounds)
    return True


def _annotate_mapped_read(custom, mapping, index, mem_node, constraints):
    """For mapped reads, apply the mapping and compute IV stride via probing.

    Stores the result in ``meta["iv_stride"]`` and transforms the index
    to physical coordinates, clearing the mapping.  Codegen linearises
    per-dimension at the MLIR level via ``_try_iv_split_offset``.

    Returns True if the IV stride was successfully computed.
    """
    symbolic_shape = custom.type.symbolic_shape
    mem_sym_shape = get_custom(mem_node).type.symbolic_shape
    idxc = IndexingContext.current()
    phys_strides = strides_from_symbolic_shape(
        idxc, mem_sym_shape, allow_mixed_shapes=True
    )

    iv_stride = compute_iv_stride_through_mapping(
        mapping, symbolic_shape, index,
        is_read=True, mem_strides=list(phys_strides),
        constraints=constraints,
    )
    if iv_stride is not None:
        custom.fx_node.meta["iv_stride"] = iv_stride
        return True
    return False


def flatten_read_indices(
    trace: CapturedTrace,
    constraints: Sequence[Constraint] = (),
):
    """Flatten N-D read indices to 1-D LINEAR_INDEX for all Read/GatherToLDS."""
    idxc = IndexingContext.current()
    div_fwd, div_bwd = get_divisibility_subs(constraints)

    for node in trace.walk(
        lambda n: isinstance(get_custom(n), (Read, GatherToLDS))
    ):
        custom = get_custom(node)

        is_g2l = isinstance(custom, GatherToLDS)
        if is_g2l:
            mapping = custom.src_mapping
            index = custom.src_index
            mem_node = custom.src
            bounds = custom.src_bounds
            dyn_vals = custom.src_mapping_dynamic_vals
        else:
            mapping = custom.mapping
            index = custom.index
            mem_node = custom.memory
            bounds = custom.bounds
            dyn_vals = custom.mapping_dynamic_vals

        if is_flattened_index(index):
            continue

        if dyn_vals:
            continue

        memory = get_custom(mem_node)
        if (
            hasattr(memory, "type")
            and hasattr(memory.type, "address_space")
            and subs_idxc(memory.type.address_space) == SHARED_ADDRESS_SPACE
        ):
            continue

        has_mapping = mapping is not None and not (
            hasattr(custom, "has_identity_mapping") and custom.has_identity_mapping()
        )

        if has_mapping:
            if not is_g2l:
                _annotate_mapped_read(
                    custom, mapping, index, mem_node, constraints
                )
            else:
                _annotate_mapped_read(
                    custom, mapping, index, mem_node, constraints
                )
        else:
            if is_g2l:
                continue
            _flatten_unmapped_read(
                custom, index, mem_node, bounds, idxc, div_fwd, div_bwd
            )
