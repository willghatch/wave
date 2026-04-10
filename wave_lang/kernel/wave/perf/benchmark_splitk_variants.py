#!/usr/bin/env python3
"""Benchmark split-K MXFP4 variants (atomic, multibuffer, MBSK) vs baseline.

Compares:
  - baseline: get_tagged_mxfp4_gemm (S=1)
  - atomic: get_tagged_splitk_mxfp4_gemm
  - multibuffer: get_tagged_multibuffer_splitk_mxfp4_gemm (main + reduction)
  - mbsk: get_tagged_mbsk_splitk_mxfp4_gemm

Usage (from worktree root with WAVE_DIR and deps set):
    python wave_lang/kernel/wave/perf/benchmark_splitk_variants.py [--csv PATH]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
from typing import Callable

import torch

from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.constraints import ScaledMMAType
from wave_lang.kernel.wave.schedules import get_mxfp4_dbuf_schedule
from wave_lang.kernel.wave.templates.tagged_mxfp4_gemm import (
    get_tagged_mbsk_splitk_mxfp4_gemm,
    get_tagged_multibuffer_splitk_mxfp4_gemm,
    get_tagged_mxfp4_gemm,
    get_tagged_splitk_mxfp4_gemm,
)
from wave_lang.kernel.wave.utils.mxfp_utils import generate_gemm_afp4wfp4_inputs
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.torch_utils import device_zeros

BLOCK_SHAPE = (128, 128, 128)

# (M, N, K), num_splits for split-K variants; baseline always S=1.
CONFIGS: list[tuple[tuple[int, int, int], int]] = [
    ((128, 128, 65536), 8),
    ((128, 128, 32768), 4),
    ((256, 256, 8192), 2),
    ((256, 256, 32768), 4),
    ((512, 512, 8192), 2),
    ((1024, 1024, 8192), 2),
    ((256, 256, 8192), 4),
    ((512, 512, 16384), 2),
    ((128, 128, 65536), 4),
    ((256, 256, 512), 2),
]


def get_flops(m: int, n: int, k: int) -> float:
    return 2.0 * m * n * k


def benchmark_kernel(run_fn: Callable[[], None], warmup: int = 3, iters: int = 10) -> float:
    """Return mean runtime in microseconds (trimmed mean)."""
    for _ in range(warmup):
        run_fn()
    torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)

    times.sort()
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    return sum(trimmed) / len(trimmed)


def compile_and_bench_baseline(shape: tuple[int, int, int], block_shape: tuple[int, int, int]) -> float:
    gemm, options = get_tagged_mxfp4_gemm(shape, block_shape)
    options = set_default_run_config(options)
    schedule = get_mxfp4_dbuf_schedule(use_stagger=True, k_partitions=1)
    compiled = wave_compile(options, gemm, schedule)

    m, n, k = shape
    x, w, x_s, w_s = generate_gemm_afp4wfp4_inputs(shape, device=torch.device("cuda"))
    w_t = w.T.contiguous()

    def run() -> None:
        c = device_zeros(m, n, dtype=torch.float32)
        compiled(x, x_s, w_t, w_s, c)

    return benchmark_kernel(run)


def compile_and_bench_atomic(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    num_splits: int,
) -> float:
    gemm, options = get_tagged_splitk_mxfp4_gemm(
        shape,
        num_splits=num_splits,
        block_shape=block_shape,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    options = set_default_run_config(options)
    schedule = get_mxfp4_dbuf_schedule(use_stagger=True, k_partitions=1)
    compiled = wave_compile(options, gemm, schedule)

    m, n, k = shape
    x, w, x_s, w_s = generate_gemm_afp4wfp4_inputs(shape, device=torch.device("cuda"))
    w_t = w.T.contiguous()

    def run() -> None:
        c = device_zeros(m, n, dtype=torch.float32)
        compiled(x, x_s, w_t, w_s, c)

    return benchmark_kernel(run)


def compile_and_bench_multibuffer(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    num_splits: int,
) -> float:
    main_fn, main_options, red_fn, red_options = get_tagged_multibuffer_splitk_mxfp4_gemm(
        shape,
        num_splits=num_splits,
        block_shape=block_shape,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    main_options = set_default_run_config(main_options)
    red_options = set_default_run_config(red_options)
    schedule = get_mxfp4_dbuf_schedule(use_stagger=True, k_partitions=1)
    compiled_main = wave_compile(main_options, main_fn, schedule)
    compiled_red = wave_compile(red_options, red_fn)

    m, n, k = shape
    x, w, x_s, w_s = generate_gemm_afp4wfp4_inputs(shape, device=torch.device("cuda"))
    w_t = w.T.contiguous()

    def run() -> None:
        workspace = device_zeros(num_splits, m, n, dtype=torch.float32)
        c = device_zeros(m, n, dtype=torch.float32)
        compiled_main(x, x_s, w_t, w_s, workspace)
        compiled_red(workspace, c)

    return benchmark_kernel(run)


def compile_and_bench_mbsk(
    shape: tuple[int, int, int],
    block_shape: tuple[int, int, int],
    num_splits: int,
) -> float:
    gemm, options = get_tagged_mbsk_splitk_mxfp4_gemm(
        shape,
        num_splits=num_splits,
        block_shape=block_shape,
        mfma_variant=ScaledMMAType.F32_16x16x128_F8F6F4,
    )
    options = set_default_run_config(options)
    schedule = get_mxfp4_dbuf_schedule(use_stagger=True, k_partitions=1)
    compiled = wave_compile(options, gemm, schedule)

    m, n, k = shape
    block_m, block_n, _ = block_shape
    num_tiles = math.ceil(m / block_m) * math.ceil(n / block_n)
    x, w, x_s, w_s = generate_gemm_afp4wfp4_inputs(shape, device=torch.device("cuda"))
    w_t = w.T.contiguous()

    def run() -> None:
        workspace = device_zeros(num_splits, m, n, dtype=torch.float32)
        sync_buf = device_zeros(num_tiles, dtype=torch.int32)
        c = device_zeros(m, n, dtype=torch.float32)
        compiled(x, x_s, w_t, w_s, workspace, sync_buf, c)

    return benchmark_kernel(run)


def runtime_to_tflops(runtime_us: float, flops: float) -> float:
    if runtime_us <= 0:
        return float("nan")
    return flops / (runtime_us * 1e-6) / 1e12


def safe_bench(label: str, fn: Callable[[], float]) -> tuple[float | None, str]:
    try:
        t = fn()
        return t, "ok"
    except Exception:
        print(f"  [{label}] FAILED:\n{traceback.format_exc()}", file=sys.stderr)
        return None, "FAILED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=str,
        default="splitk_variant_benchmark_results.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    fieldnames = [
        "m",
        "n",
        "k",
        "split_S",
        "variant",
        "runtime_us",
        "tflops",
        "status",
    ]

    for shape, num_splits in CONFIGS:
        m, n, k = shape
        flops = get_flops(m, n, k)
        print(f"\n=== Shape ({m}, {n}, {k}), split S={num_splits} ===", flush=True)

        variants: list[tuple[str, Callable[[], float]]] = [
            ("baseline", lambda: compile_and_bench_baseline(shape, BLOCK_SHAPE)),
            ("atomic", lambda: compile_and_bench_atomic(shape, BLOCK_SHAPE, num_splits)),
            (
                "multibuffer",
                lambda: compile_and_bench_multibuffer(shape, BLOCK_SHAPE, num_splits),
            ),
            ("mbsk", lambda: compile_and_bench_mbsk(shape, BLOCK_SHAPE, num_splits)),
        ]

        for name, bench_fn in variants:
            rt, status = safe_bench(name, bench_fn)
            split_s = 1 if name == "baseline" else num_splits
            if rt is None:
                row = {
                    "m": m,
                    "n": n,
                    "k": k,
                    "split_S": split_s,
                    "variant": name,
                    "runtime_us": "FAILED",
                    "tflops": "FAILED",
                    "status": status,
                }
                print(f"  {name}: FAILED", flush=True)
            else:
                tf = runtime_to_tflops(rt, flops)
                row = {
                    "m": m,
                    "n": n,
                    "k": k,
                    "split_S": split_s,
                    "variant": name,
                    "runtime_us": f"{rt:.2f}",
                    "tflops": f"{tf:.4f}",
                    "status": status,
                }
                print(f"  {name}: {rt:.2f} us, {tf:.4f} TFLOPS", flush=True)
            rows.append(row)

    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
