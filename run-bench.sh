#!/usr/bin/env bash
# Quick launcher for the 128x256 preshuffle-B MXFP4 GEMM benchmark.
#
# Usage:
#   ./run-bench.sh                        # 4-wave (default)
#   ./run-bench.sh --wave-shape 2,4       # 8-wave
#   ./run-bench.sh -o my_results.csv      # custom output file
#
# Any extra arguments are forwarded to benchmark_mxfp4.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python -u "$SCRIPT_DIR/wave_lang/kernel/wave/perf/benchmark_mxfp4.py" \
    --template mxfp4_preshuffle_b \
    --wave-shape 1,4 \
    --unroll-factor 6 \
    --shapes "$SCRIPT_DIR/wave_lang/kernel/wave/perf/mxfp4_128x256_shapes.csv" \
    --warmup-iters 5 \
    --benchmark-iters 20 \
    -o results_128x256.csv \
    "$@"
