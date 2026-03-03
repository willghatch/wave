# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Standalone reproducible benchmark for MXFP4 Wave GEMM.

Uses rocprof for benchmarking, invoking kernels with wave runtime.

Shapes from CSV with required columns M, N, K, MT_M, MT_N, and MT_K:

  python -u wave_lang/kernel/wave/perf/benchmark_mxfp4.py --shapes wave_lang/kernel/wave/perf/mxfp4_shapes_macrotiles.csv -o results.csv

Preshuffle-B template (128x256 kernel from examples/python/7.x_128x256-gemm.py):

  python -u wave_lang/kernel/wave/perf/benchmark_mxfp4.py \\
      --template mxfp4_preshuffle_b \\
      --wave-shape 1,4 --unroll-factor 6 \\
      --shapes wave_lang/kernel/wave/perf/mxfp4_128x256_shapes.csv \\
      -o results.csv

Optional env: ATT_LIBRARY_PATH=/path/to/rocprof-trace-decoder/rocm/lib. If set, passed to rocprofv3 (--att --att-library-path).
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from wave_lang.kernel.wave.compile import wave_compile
from wave_lang.kernel.wave.schedules import (
    get_mxfp4_dbuf_schedule,
    get_mxfp4_preshuffle_b_schedule,
)
from wave_lang.kernel.wave.templates.tagged_mxfp4_gemm import (
    get_tagged_mxfp4_gemm,
    get_tagged_mxfp4_gemm_preshuffle_b,
)
from wave_lang.kernel.lang.global_symbols import *
from wave_lang.kernel.wave.utils.run_utils import set_default_run_config
from wave_lang.kernel.wave.utils.mxfp_utils import (
    generate_gemm_afp4wfp4_inputs,
    torchScaledGemmMXFP4,
    b_preshuffle,
    e8m0_shuffle,
)
from wave_lang.kernel.wave.perf.utils import (
    find_rocprof_outputs,
    get_rocprofv3_cmd,
    parse_rocprof_kernel_stats,
)

# ---------------------------------------------------------------------------
# Wave kernel template and compile options
# ---------------------------------------------------------------------------


def get_mxfp4_gemm_wave(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
):
    gemm, options = get_tagged_mxfp4_gemm(shape, macrotiles)
    schedule = get_mxfp4_dbuf_schedule(use_stagger=True)
    options = set_default_run_config(options)

    compiled_gemm = wave_compile(options, gemm, schedule)
    return compiled_gemm


_PRESHUFFLE_B_PIPELINE_STAGES = 3
_PRESHUFFLE_B_PEELED_ITERS = _PRESHUFFLE_B_PIPELINE_STAGES - 1  # 2 (prologue only)


def _pick_unroll_factor(kernel_iters: int, preferred: int) -> int:
    """Return the largest factor of kernel_iters that is <= preferred.

    Falls back to 1 (no unrolling) if nothing else divides evenly.
    """
    for f in range(preferred, 1, -1):
        if kernel_iters % f == 0:
            return f
    return 1


def get_mxfp4_preshuffle_b_gemm_wave(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    wave_shape: tuple[int, int] = (1, 4),
    unroll_factor: int = 6,
):
    """Compile the preshuffle-B MXFP4 GEMM (examples/python/7.x_128x256-gemm.py).

    The 3-stage pipeline peels ``num_stages - 1 = 2`` prologue iterations from
    the KERNEL loop count (the epilogue drains after the loop ends without
    reducing its count).  So the KERNEL loop body runs for
    ``K / BLOCK_K - 2`` iterations.  The unroll factor must divide that count
    evenly.  If the requested unroll_factor does not divide, we fall back to
    the largest factor <= the requested value.
    """
    _M, _N, K = shape
    _MT_M, _MT_N, BLOCK_K = macrotiles
    total_iters = K // BLOCK_K
    kernel_iters = total_iters - _PRESHUFFLE_B_PEELED_ITERS
    effective_unroll = _pick_unroll_factor(kernel_iters, unroll_factor)
    if effective_unroll != unroll_factor:
        print(
            f"  Note: unroll_factor {unroll_factor} does not divide "
            f"kernel_iters={kernel_iters} (K/BLOCK_K={total_iters} - "
            f"{_PRESHUFFLE_B_PEELED_ITERS} peeled); using {effective_unroll}"
        )

    gemm, options = get_tagged_mxfp4_gemm_preshuffle_b(
        shape, macrotiles, wave_shape=wave_shape
    )
    options.minimize_shared_allocs = True
    options.linearize_shared_access = True
    options.use_buffer_ops = True
    schedule = get_mxfp4_preshuffle_b_schedule(unroll_factor=effective_unroll)
    options = set_default_run_config(options)
    compiled_gemm = wave_compile(options, gemm, schedule)
    return compiled_gemm


# ---------------------------------------------------------------------------
# Helpers (IREE/rocprof, validate, benchmark)
# ---------------------------------------------------------------------------


class BenchmarkError(RuntimeError):
    """Raised when a benchmark step fails (compile, validation, or benchmark run)."""

    def __init__(self, message: str, stage: str):
        super().__init__(message)
        self.stage = (
            stage  # "compile_failed", "validation_failed", or "benchmark_failed"
        )


def get_flops(M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K


def get_byte_count_mxfp4(M: int, N: int, K: int) -> int:
    # Packed mxfp4: K/2 bytes per row for A (M rows), B (N rows)
    # Scales: K/32 per row for A and B. Output: M*N*2 (bf16).
    return M * (K // 2) + M * (K // 32) + N * (K // 2) + N * (K // 32) + M * N * 2


def runtime_us_to_tflops(M: int, N: int, K: int, runtime_us: float) -> float:
    if runtime_us <= 0:
        return 0.0
    flops = get_flops(M, N, K)
    return (flops / 1e12) / (runtime_us / 1e6)


# --- rocprof / worker helpers -------------------------------------------------


def _clear_dir(dir_path: os.PathLike) -> None:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def _parse_worker_stdout_for_mean_us(stdout: str) -> tuple[float, bool]:
    """Parse worker stdout for a line 'MEAN_US: <float>'; return (mean_us, ok)."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("MEAN_US:"):
            try:
                value = float(line.split(":", 1)[1].strip())
                return value, True
            except (ValueError, IndexError):
                pass
    return 0.0, False


def _run_torch_benchmark(
    kernel_func,
    inputs: tuple,
    warmup_iters: int = 0,
    benchmark_iters: int = 1,
) -> float:
    """Run warmup and benchmark loop with torch.cuda.synchronize; return mean runtime in microseconds."""
    for _ in range(warmup_iters):
        kernel_func(*inputs)
    torch.cuda.synchronize()

    if benchmark_iters < 1:
        return 0.0
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(benchmark_iters):
        kernel_func(*inputs)
    end_ev.record()
    torch.cuda.synchronize()
    mean_ms = start_ev.elapsed_time(end_ev) / benchmark_iters
    return mean_ms * 1000.0  # us


def run_worker(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    warmup_iters: int = 0,
    benchmark_iters: int = 1,
    template: str = "mxfp4",
    wave_shape: tuple[int, int] = (1, 4),
    unroll_factor: int = 6,
) -> None:
    """Worker entry: compile GEMM with wave_runtime, run torch benchmark, print MEAN_US to stdout."""
    m, n, k = shape
    device = torch.device("cuda")

    if template == "mxfp4_preshuffle_b":
        gemm_rt = get_mxfp4_preshuffle_b_gemm_wave(
            shape, macrotiles, wave_shape=wave_shape, unroll_factor=unroll_factor
        )
        x, w, x_scale, w_scale = generate_gemm_afp4wfp4_inputs((m, n, k), device)
        w_t = w.T.contiguous()
        w_t_ps = b_preshuffle(w_t)
        x_scale_ps = e8m0_shuffle(x_scale)
        w_scale_ps = e8m0_shuffle(w_scale)
        wave_out = torch.empty(m, w_t_ps.shape[0], device=device, dtype=torch.float32)
        inputs = (x, x_scale_ps, w_t_ps, w_scale_ps, wave_out)
    else:
        gemm_rt = get_mxfp4_gemm_wave(shape, macrotiles)
        x, w, x_scale, w_scale = generate_gemm_afp4wfp4_inputs((m, n, k), device)
        w_t = w.T.contiguous()
        wave_out = torch.empty(m, n, device=device, dtype=torch.float32)
        inputs = (x, x_scale, w_t, w_scale, wave_out)

    mean_us = _run_torch_benchmark(
        gemm_rt, inputs, warmup_iters=warmup_iters, benchmark_iters=benchmark_iters
    )
    print(f"MEAN_US: {mean_us}")


def validate_mxfp4_gemm(
    shape: tuple[int, int, int],
    compiled_gemm,
    template: str = "mxfp4",
) -> bool:
    """Run compiled Wave GEMM (with wave_runtime=True) and compare to torch reference."""
    m, n, k = shape
    try:
        device = torch.device("cuda")
        x, w, x_scale, w_scale = generate_gemm_afp4wfp4_inputs(shape, device)
        torch_ref = torchScaledGemmMXFP4(x, w, x_scale, w_scale)

        if template == "mxfp4_preshuffle_b":
            w_t = w.T.contiguous()
            w_t_ps = b_preshuffle(w_t)
            x_scale_ps = e8m0_shuffle(x_scale)
            w_scale_ps = e8m0_shuffle(w_scale)
            wave_out = torch.zeros(
                m, w_t_ps.shape[0], device=device, dtype=torch.float32
            )
            compiled_gemm(x, x_scale_ps, w_t_ps, w_scale_ps, wave_out)
        else:
            w_t = w.T.contiguous()
            wave_out = torch.zeros(m, n, device=device, dtype=torch.float32)
            compiled_gemm(x, x_scale, w_t, w_scale, wave_out)

        torch.testing.assert_close(
            wave_out, torch_ref, check_dtype=False, check_device=False
        )
        return True
    except Exception as e:
        raise RuntimeError(f"Validation failed for shape {shape}: {e}") from e


def benchmark_mxfp4_gemm_rocprof(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    profiler_dump_path: Path,
    att_library_path: Optional[str],
    kernel_regex: str = "gemm",
    timeout: Optional[float] = None,
    warmup_iters: int = 0,
    benchmark_iters: int = 1,
    template: str = "mxfp4",
    wave_shape: tuple[int, int] = (1, 4),
    unroll_factor: int = 6,
) -> float:
    """Run self as worker under rocprofv3 (torch benchmark); return mean runtime in microseconds.

    Raises RuntimeError if the subprocess times out, exits non-zero, the worker does not
    output MEAN_US, or rocprof parsing fails.
    """
    m, n, k = shape
    mt_m, mt_n, mt_k = macrotiles
    _clear_dir(profiler_dump_path)
    profile_prefix = get_rocprofv3_cmd(
        profiler_dump_path, None, kernel_regex, att_library_path
    )
    worker_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker",
        "--_shape",
        str(m),
        str(n),
        str(k),
        "--_tiles",
        str(mt_m),
        str(mt_n),
        str(mt_k),
        "--warmup-iters",
        str(warmup_iters),
        "--benchmark-iters",
        str(benchmark_iters),
        "--template",
        template,
        "--wave-shape",
        f"{wave_shape[0]},{wave_shape[1]}",
        "--unroll-factor",
        str(unroll_factor),
    ]
    full_cmd = profile_prefix + worker_cmd
    try:
        proc = subprocess.run(
            full_cmd,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"rocprof worker subprocess timed out after {timeout}s for shape {shape}"
        ) from e
    if proc.returncode != 0:
        stderr_preview = (proc.stderr or "").strip()[:500]
        raise RuntimeError(
            f"rocprof worker subprocess exited with code {proc.returncode} for shape {shape}"
            + (f"; stderr: {stderr_preview}" if stderr_preview else "")
        )
    _, worker_ok = _parse_worker_stdout_for_mean_us(proc.stdout)
    if not worker_ok:
        raise RuntimeError(
            f"Torch benchmarking failed within worker for shape {shape}: no MEAN_US in worker stdout"
        )
    stats_path, _ = find_rocprof_outputs(profiler_dump_path, None)
    if stats_path is None:
        raise RuntimeError(
            f"rocprof kernel stats file not found under {profiler_dump_path} for shape {shape}"
        )
    rocprof_stats = parse_rocprof_kernel_stats(stats_path, kernel_regex)
    if not rocprof_stats or "mean_duration_us" not in rocprof_stats:
        raise RuntimeError(
            f"rocprof kernel stats parsing failed for regex {kernel_regex!r} in {stats_path} for shape {shape}"
        )
    return rocprof_stats["mean_duration_us"]


def run_validate_and_benchmark(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
    dump_dir: Path,
    att_library_path: Optional[str],
    kernel_regex: str = "gemm",
    warmup_iters: int = 0,
    benchmark_iters: int = 1,
    template: str = "mxfp4",
    wave_shape: tuple[int, int] = (1, 4),
    unroll_factor: int = 6,
) -> tuple[Optional[float], Optional[float], str]:
    """
    Compile with wave_runtime and validate; then benchmark via torch worker under rocprofv3.
    Returns (runtime_us, tflops, status) where status is "ok", "compile_failed", "validation_failed",
    or "benchmark_failed".
    """
    m, n, k = shape
    mt_m, mt_n, mt_k = macrotiles
    gemm_id = f"gemm_{m}_{n}_{k}_MT_{mt_m}_{mt_n}_{mt_k}"

    # Compile for validation (wave_runtime=True)
    try:
        if template == "mxfp4_preshuffle_b":
            gemm_rt = get_mxfp4_preshuffle_b_gemm_wave(
                shape, macrotiles, wave_shape=wave_shape, unroll_factor=unroll_factor
            )
        else:
            gemm_rt = get_mxfp4_gemm_wave(shape, macrotiles)
    except Exception as e:
        raise BenchmarkError(
            f"Compilation failed for shape {shape}: {e}", stage="compile_failed"
        ) from e

    # Save MLIR to dump directory
    mlir_dir = dump_dir / "mlir"
    mlir_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = mlir_dir / f"{gemm_id}.mlir"
    mlir_path.write_text(gemm_rt.asm)

    # Validate numerics
    try:
        validate_mxfp4_gemm(shape, gemm_rt, template=template)
    except Exception as e:
        raise BenchmarkError(
            f"Validation failed for shape {shape}: {e}", stage="validation_failed"
        ) from e

    # Benchmark via torch worker under rocprofv3
    profiler_dump_path = dump_dir / "rocprof" / gemm_id
    try:
        runtime_us = benchmark_mxfp4_gemm_rocprof(
            shape,
            macrotiles,
            profiler_dump_path,
            att_library_path,
            kernel_regex=kernel_regex,
            warmup_iters=warmup_iters,
            benchmark_iters=benchmark_iters,
            template=template,
            wave_shape=wave_shape,
            unroll_factor=unroll_factor,
        )
    except RuntimeError as e:
        raise BenchmarkError(str(e), stage="benchmark_failed") from e

    tflops = runtime_us_to_tflops(m, n, k, runtime_us)
    return runtime_us, tflops, "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_wave_shape(s: str) -> tuple[int, int]:
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"wave-shape must be two comma-separated ints, got {s!r}"
        )
    return (int(parts[0]), int(parts[1]))


def parse_args():
    p = argparse.ArgumentParser(
        description="Standalone MXFP4 Wave GEMM benchmark (no kernel_bench)."
    )
    p.add_argument(
        "--_worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--_shape",
        type=int,
        nargs=3,
        metavar=("M", "N", "K"),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--_tiles",
        type=int,
        nargs=3,
        metavar=("mt_m", "mt_n", "mt_k"),
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--shapes",
        type=Path,
        metavar="CSV",
        help="CSV path with header M,N,K (optional columns MT_M, MT_N, MT_K for macrotile sizes)",
    )
    p.add_argument(
        "--template",
        type=str,
        choices=["mxfp4", "mxfp4_preshuffle_b"],
        default="mxfp4",
        help="Kernel template: 'mxfp4' (default, non-preshuffle) or "
        "'mxfp4_preshuffle_b' (128x256 preshuffle-B kernel)",
    )
    p.add_argument(
        "--wave-shape",
        type=_parse_wave_shape,
        default=(1, 4),
        metavar="M,N",
        help="Wave shape as M,N (default: 1,4 for 4-wave). "
        "Only used with --template mxfp4_preshuffle_b",
    )
    p.add_argument(
        "--unroll-factor",
        type=int,
        default=6,
        help="K-loop unroll factor (default: 6). "
        "Only used with --template mxfp4_preshuffle_b",
    )
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=0,
        metavar="N",
        help="Warmup iterations for benchmark (default: 0)",
    )
    p.add_argument(
        "--benchmark-iters",
        type=int,
        default=1,
        metavar="N",
        help="Benchmark iterations (default: 1)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV for results (required)",
    )
    p.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("/tmp/bench_mxfp4_dump"),
        help="Directory for rocprof output (default: /tmp/bench_mxfp4_dump)",
    )
    p.add_argument(
        "--kernel-regex",
        type=str,
        default="gemm",
        help="Regex for rocprof kernel filter (default: gemm)",
    )
    return p.parse_args()


def validate_shape_and_macrotiles(
    shape: tuple[int, int, int],
    macrotiles: tuple[int, int, int],
) -> None:
    """Validate shape and macrotile combination. Raises ValueError with a reason if invalid."""
    M, N, K = shape
    mt_m, mt_n, mt_k = macrotiles
    if M <= 4 or N <= 4 or K <= 4:
        raise ValueError(f"M, N, K must be > 4 (got M={M}, N={N}, K={K})")
    if mt_m > M or mt_n > N or mt_k > K:
        raise ValueError(
            f"Macrotiles must not exceed shape dimensions: "
            f"MT_M({mt_m})<=M({M}), MT_N({mt_n})<=N({N}), MT_K({mt_k})<=K({K})"
        )
    if K % 32 != 0:
        raise ValueError(f"K must be divisible by 32 for scale matrix size (got K={K})")
    if mt_m % 4 != 0:
        raise ValueError(
            f"MT_M must be divisible by 4 (wave constraint) (got MT_M={mt_m})"
        )
    if mt_n % 2 != 0:
        raise ValueError(
            f"MT_N must be divisible by 2 (wave constraint) (got MT_N={mt_n})"
        )


def load_shapes_csv(
    path: Path,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Load shape (M,N,K) and macrotile sizes (MT_M, MT_N, MT_K) from CSV.
    Validates each row with validate_shape_and_macrotiles; raises ValueError on first invalid row.
    """
    rows: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader, start=2):  # 2 = header is row 1
            M = int(row["M"])
            N = int(row["N"])
            K = int(row["K"])
            MT_M = int(row["MT_M"])
            MT_N = int(row["MT_N"])
            MT_K = int(row["MT_K"])
            shape = (M, N, K)
            macrotiles = (MT_M, MT_N, MT_K)
            try:
                validate_shape_and_macrotiles(shape, macrotiles)
            except ValueError as e:
                raise ValueError(f"{path}: row {row_idx}: {e}") from e
            rows.append((shape, macrotiles))
    return rows


def main():
    att_library_path = os.environ.get("ATT_LIBRARY_PATH") or None

    args = parse_args()

    if args._worker:
        if args._shape is None:
            print("--_worker requires --_shape M N K.", file=sys.stderr)
            sys.exit(1)
        if args._tiles is None:
            print("--_worker requires --_tiles mt_m mt_n mt_k.", file=sys.stderr)
            sys.exit(1)
        shape = tuple(args._shape)
        macrotiles = tuple(args._tiles)
        try:
            validate_shape_and_macrotiles(shape, macrotiles)
        except ValueError as e:
            print(f"Invalid shape/macrotile: {e}", file=sys.stderr)
            sys.exit(1)
        run_worker(
            shape,
            macrotiles,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            template=args.template,
            wave_shape=args.wave_shape,
            unroll_factor=args.unroll_factor,
        )
        return

    if args.shapes is None:
        print("--shapes <path/to/csv> is required.", file=sys.stderr)
        sys.exit(1)

    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dump_dir = dump_dir / run_id
    run_dump_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dump directory for this run: {run_dump_dir}")
    kernel_regex = args.kernel_regex
    warmup_iters = args.warmup_iters
    benchmark_iters = args.benchmark_iters
    template = args.template
    wave_shape = args.wave_shape
    unroll_factor = args.unroll_factor

    if template == "mxfp4_preshuffle_b":
        print(
            f"Template: mxfp4_preshuffle_b, "
            f"wave_shape={wave_shape}, unroll_factor={unroll_factor}"
        )

    # --shapes mode
    if not args.shapes.exists():
        print(f"Shapes file not found: {args.shapes}", file=sys.stderr)
        sys.exit(1)
    if args.output is None:
        print("--shapes requires -o/--output for the result CSV.", file=sys.stderr)
        sys.exit(1)

    try:
        shape_rows = load_shapes_csv(args.shapes)
    except ValueError as e:
        print(f"Invalid shape/macrotile in CSV: {e}", file=sys.stderr)
        sys.exit(1)
    if not shape_rows:
        print("No shapes found in CSV.", file=sys.stderr)
        sys.exit(1)

    results = []
    failed_compilation = []
    failed_validation = []
    failed_benchmark = []
    for shape, macrotiles in shape_rows:
        M, N, K = shape
        mt_m, mt_n, mt_k = macrotiles
        try:
            runtime_us, tflops, status = run_validate_and_benchmark(
                shape,
                macrotiles,
                run_dump_dir,
                att_library_path,
                kernel_regex=kernel_regex,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                template=template,
                wave_shape=wave_shape,
                unroll_factor=unroll_factor,
            )
        except BenchmarkError as e:
            print(f"  {e}", file=sys.stderr)
            traceback.print_exc()
            status = e.stage
            runtime_us, tflops = None, None
        ok = status == "ok"
        if status == "compile_failed":
            failed_compilation.append((M, N, K))
        elif status == "validation_failed":
            failed_validation.append((M, N, K))
        elif status == "benchmark_failed":
            failed_benchmark.append((M, N, K))
        mean_us = runtime_us if runtime_us is not None else 0.0
        tflops_val = tflops if tflops is not None else 0.0
        results.append(
            {
                "M": M,
                "N": N,
                "K": K,
                "MT_M": mt_m,
                "MT_N": mt_n,
                "MT_K": mt_k,
                "runtime_us": mean_us,
                "tflops": tflops_val,
                "ok": ok,
            }
        )
        status_str = "ok" if ok else status
        print(
            f"  ({M}, {N}, {K}) MT({mt_m},{mt_n},{mt_k}): {mean_us:.2f} us, {tflops_val:.4f} TFLOPs [{status_str}]"
        )

    if failed_compilation:
        print(
            f"Kernels that failed compilation: {failed_compilation}",
            file=sys.stderr,
        )
    if failed_validation:
        print(
            f"Kernels that failed numerical validation: {failed_validation}",
            file=sys.stderr,
        )
    if failed_benchmark:
        print(f"Kernels that failed benchmarking: {failed_benchmark}", file=sys.stderr)

    valid = [r for r in results if r["ok"] and r["runtime_us"] > 0]
    if not valid:
        print("No successful runs.", file=sys.stderr)
        sys.exit(1)

    avg_us = sum(r["runtime_us"] for r in valid) / len(valid)
    avg_tflops = sum(r["tflops"] for r in valid) / len(valid)
    best = max(valid, key=lambda r: r["tflops"])

    print()
    print(
        f"Average across {len(valid)} kernels: {avg_us:.2f} us, {avg_tflops:.4f} TFLOPs"
    )
    print(
        f"Best kernel: M={best['M']}, N={best['N']}, K={best['K']} MT({best['MT_M']},{best['MT_N']},{best['MT_K']}) -> "
        f"{best['runtime_us']:.2f} us, {best['tflops']:.4f} TFLOPs"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "M",
                "N",
                "K",
                "MT_M",
                "MT_N",
                "MT_K",
                "runtime_us",
                "tflops",
                "ok",
            ],
        )
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "M": r["M"],
                    "N": r["N"],
                    "K": r["K"],
                    "MT_M": r["MT_M"],
                    "MT_N": r["MT_N"],
                    "MT_K": r["MT_K"],
                    "runtime_us": r["runtime_us"],
                    "tflops": r["tflops"],
                    "ok": r["ok"],
                }
            )
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
