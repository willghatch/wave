#!/usr/bin/env python3
"""Benchmark Wave MXFP4 preshuffle-B GEMM (4-wave) vs aiter gemm_a4w4.

Uses the same compilation pipeline as benchmark_mxfp4.py (called by
run-bench.sh) with wave_runtime=True so that both scripts produce the
same Wave kernel binary.

Usage:
    python run-7.x-vs-aiter-bench.py --shapes-csv wave_lang/kernel/wave/perf/mxfp4_128x256_shapes.csv
    python run-7.x-vs-aiter-bench.py --shapes 1024,1024,8192 2048,2048,8192
    python run-7.x-vs-aiter-bench.py --shapes-csv shapes.csv --iters 50 --warmup 10
    python run-7.x-vs-aiter-bench.py --skip-aiter
    python run-7.x-vs-aiter-bench.py --skip-wave
"""

import argparse
import csv
import os
import sys
import time
import traceback

os.environ["WAVE_CACHE_ON"] = "0"
os.environ.setdefault("AITER_JIT_DIR", os.path.expanduser("~/.aiter/jit"))

import torch

DEFAULT_BLOCK = (128, 256, 256)
DEFAULT_WAVE_SHAPE = (1, 4)
DEFAULT_UNROLL_FACTOR = 6

_PRESHUFFLE_B_PIPELINE_STAGES = 3
_PRESHUFFLE_B_PEELED_ITERS = _PRESHUFFLE_B_PIPELINE_STAGES - 1  # 2


def _pick_unroll_factor(kernel_iters: int, preferred: int) -> int:
    """Return the largest factor of kernel_iters that is <= preferred."""
    for f in range(preferred, 1, -1):
        if kernel_iters % f == 0:
            return f
    return 1


def bench_cuda_events(fn, warmup_iters, bench_iters):
    """Time fn() using CUDA events.  Returns mean microseconds."""
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(bench_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / bench_iters


def calc_tflops(M, N, K, us):
    return 2.0 * M * N * K / us / 1e6


def parse_shape(s):
    parts = s.split(",")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def load_shapes_csv(path):
    """Load shapes from CSV.  Returns list of (shape, block) tuples."""
    entries = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            shape = (int(row["M"]), int(row["N"]), int(row["K"]))
            block = (int(row["MT_M"]), int(row["MT_N"]), int(row["MT_K"]))
            entries.append((shape, block))
    return entries


def compile_wave_kernel(shape, macrotiles, wave_shape, unroll_factor):
    """Compile via wave_compile -- same path as benchmark_mxfp4.py.

    Returns a callable WaveKernel or None on failure.
    """
    from wave_lang.kernel.wave.compile import wave_compile
    from wave_lang.kernel.wave.schedules import get_mxfp4_preshuffle_b_schedule
    from wave_lang.kernel.wave.templates import get_tagged_mxfp4_gemm_preshuffle_b
    from wave_lang.kernel.wave.utils.run_utils import set_default_run_config

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
    options.wave_runtime = True

    return wave_compile(options, gemm, schedule)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Wave MXFP4 preshuffle-B GEMM vs aiter"
    )
    parser.add_argument(
        "--shapes-csv", type=str, default=None,
        help="CSV file with columns M, N, K, MT_M, MT_N, MT_K",
    )
    parser.add_argument(
        "--shapes", nargs="+", default=None,
        help="M,N,K shapes (uses block 128,256,256; ignored if --shapes-csv given)",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--skip-aiter", action="store_true")
    parser.add_argument("--skip-wave", action="store_true")
    args = parser.parse_args()

    if args.shapes_csv:
        entries = load_shapes_csv(args.shapes_csv)
    elif args.shapes:
        entries = [(parse_shape(s), DEFAULT_BLOCK) for s in args.shapes]
    else:
        entries = [((1024, 1024, 8192), DEFAULT_BLOCK)]

    warmup = args.warmup
    iters = args.iters

    print(f"Shapes: {len(entries)}, Warmup: {warmup}, Iterations: {iters}")

    # ------------------------------------------------------------------
    # Compile Wave kernels (compilation is slow, do it once per config)
    # ------------------------------------------------------------------

    wave_kernels = {}

    if not args.skip_wave:
        for idx, (shape, block) in enumerate(entries):
            label = "wave-4w"
            key = (shape, block, label)
            if key in wave_kernels:
                continue
            print(
                f"[{idx+1}/{len(entries)}] Compiling {label} "
                f"shape={shape} block={block} [wave_runtime]...",
                flush=True,
            )
            t0 = time.time()
            try:
                compiled = compile_wave_kernel(
                    shape, block,
                    wave_shape=DEFAULT_WAVE_SHAPE,
                    unroll_factor=DEFAULT_UNROLL_FACTOR,
                )
                elapsed = time.time() - t0
                print(f"  compiled in {elapsed:.1f}s", flush=True)
                wave_kernels[key] = compiled
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  FAILED in {elapsed:.1f}s: {e}", flush=True)
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Check aiter availability
    # ------------------------------------------------------------------

    aiter_mod = None
    if not args.skip_aiter:
        try:
            import aiter
            from aiter import dtypes  # noqa: F401
            from aiter.ops.shuffle import shuffle_weight  # noqa: F401

            aiter_mod = aiter
        except Exception as e:
            print(f"WARNING: aiter not available, skipping: {e}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------

    results = []

    for idx, (shape, block) in enumerate(entries):
        M, N, K = shape
        print(f"\n{'='*70}")
        print(
            f"[{idx+1}/{len(entries)}] "
            f"Shape: M={M}, N={N}, K={K}   block={block}"
        )
        print(f"Warmup={warmup}, Iterations={iters}")
        print(f"{'='*70}")

        # --- Wave kernel ---
        if not args.skip_wave:
            from wave_lang.kernel.wave.utils.mxfp_utils import (
                generate_gemm_afp4wfp4_inputs,
                b_preshuffle,
                e8m0_shuffle,
            )

            label = "wave-4w"
            key = (shape, block, label)
            if key not in wave_kernels:
                print(f"  {label:>12s}:  SKIPPED (compilation failed)")
            else:
                compiled_gemm = wave_kernels[key]
                device = torch.device("cuda")
                x, w, x_scale, w_scale = generate_gemm_afp4wfp4_inputs(shape, device)
                w_t = w.T.contiguous()
                w_t_ps = b_preshuffle(w_t)
                x_scale_ps = e8m0_shuffle(x_scale)
                w_scale_ps = e8m0_shuffle(w_scale)
                wave_out = torch.zeros(
                    M, w_t_ps.shape[0], dtype=torch.float32, device=device
                )

                def run_wave(
                    _gemm=compiled_gemm, _x=x, _xs=x_scale_ps,
                    _w=w_t_ps, _ws=w_scale_ps, _out=wave_out,
                ):
                    _gemm(_x, _xs, _w, _ws, _out)

                us = bench_cuda_events(run_wave, warmup, iters)
                tf = calc_tflops(M, N, K, us)
                results.append({
                    "shape": shape, "block": block,
                    "backend": label, "us": us, "tflops": tf,
                })
                print(f"  {label:>12s}:  {us:8.1f} us  {tf:8.2f} TFLOPS")

                del x, w_t, w_t_ps, x_scale, w_scale, x_scale_ps, w_scale_ps, wave_out
                torch.cuda.empty_cache()

        # --- aiter kernel ---
        if aiter_mod is not None:
            from aiter import dtypes
            from aiter.ops.shuffle import shuffle_weight

            dtype_bf16 = dtypes.bf16
            quant_func = aiter_mod.get_triton_quant(aiter_mod.QuantType.per_1x32)
            x_fp = torch.randn((M, K), dtype=dtype_bf16, device="cuda:0")
            w_fp = torch.randn((N, K), dtype=dtype_bf16, device="cuda:0")
            x_q, x_sc = quant_func(x_fp, shuffle=True)
            w_q, w_sc = quant_func(w_fp, shuffle=True)
            w_q = shuffle_weight(w_q)
            del x_fp, w_fp
            torch.cuda.empty_cache()

            m_pad = (M + 31) // 32 * 32
            aiter_out = torch.empty(
                (m_pad, N), dtype=dtype_bf16, device="cuda:0"
            )

            def run_aiter(
                _xq=x_q, _wq=w_q, _xsc=x_sc, _wsc=w_sc, _out=aiter_out,
            ):
                aiter_mod.gemm_a4w4(
                    _xq, _wq, _xsc, _wsc, _out, bpreshuffle=True,
                )

            us = bench_cuda_events(run_aiter, warmup, iters)
            tf = calc_tflops(M, N, K, us)
            results.append({
                "shape": shape, "block": block,
                "backend": "aiter", "us": us, "tflops": tf,
            })
            print(f"  {'aiter':>12s}:  {us:8.1f} us  {tf:8.2f} TFLOPS")

            del x_q, w_q, x_sc, w_sc, aiter_out
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")

    backends_seen = []
    for r in results:
        if r["backend"] not in backends_seen:
            backends_seen.append(r["backend"])

    header = f"{'M':>6s} {'N':>6s} {'K':>6s} {'block':>20s}"
    for b in backends_seen:
        header += f" {b + ' (TFLOPS)':>18s}"
    if "aiter" in backends_seen and len(backends_seen) > 1:
        for b in backends_seen:
            if b != "aiter":
                header += f" {b + '/aiter':>12s}"
    print(header)
    print("-" * len(header))

    for shape, block in entries:
        M, N, K = shape
        block_str = f"{block[0]}x{block[1]}x{block[2]}"
        row = f"{M:6d} {N:6d} {K:6d} {block_str:>20s}"
        shape_results = {
            r["backend"]: r for r in results
            if r["shape"] == shape and r["block"] == block
        }
        for b in backends_seen:
            if b in shape_results:
                row += f" {shape_results[b]['tflops']:18.2f}"
            else:
                row += f" {'N/A':>18s}"
        aiter_tf = shape_results.get("aiter", {}).get("tflops", 0)
        if aiter_tf > 0 and len(backends_seen) > 1:
            for b in backends_seen:
                if b != "aiter" and b in shape_results:
                    ratio = shape_results[b]["tflops"] / aiter_tf
                    row += f" {ratio:11.2f}x"
                elif b != "aiter":
                    row += f" {'N/A':>12s}"
        print(row)

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
