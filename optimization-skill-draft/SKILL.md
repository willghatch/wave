---
name: optimize-wave-kernel
description: >-
  Optimize a functional Wave GPU kernel for performance on AMD GPUs
  (CDNA3/CDNA4). Use when the user has a working kernel and wants to
  improve TFLOPS, reduce register pressure, tune schedules, or profile
  with rocprofv3/ATT. Covers benchmarking, assembly analysis, tracing,
  schedule parameter tuning, and iterative performance experiments.
---

# Optimize Wave Kernel

Guide for taking a **functional** Wave kernel to higher performance on AMD CDNA GPUs.
Assumes the kernel compiles, runs, and passes numerical validation.

## Repo documentation references

Read these files from the Wave repo for detailed background:

- `docs/wave/asm_backend.rst` -- waveasm compilation pipeline, register allocation, peephole optimizations, debugging env vars
- `docs/wave/trace.rst` -- ATT thread trace capture with rocprofv3
- `docs/wave/wave_schedule.rst` -- the `wave_schedule` construct for explicit graph manipulation, pipelining, and tag-based scheduling
- `docs/wave/four_stage_scheduler.rst` -- four-stage pipelined scheduler (global load, local store, local load, compute)
- `docs/wave/multibuffering.rst` -- index-based multibuffering for shared-memory hazard elimination
- `docs/wave/schedule.rst` -- schedule file format (initiation interval, resource reservation table, stage assignment)
- `docs/wave/optimize_schedule.rst` -- automated schedule optimization with hill climbing (`tune_attention`)
- `benchmark/wave_att_analyzer.py` -- script to compute MFMA/XDL utilization from ATT traces

## Phase 0: Establish a baseline

Before changing anything, capture reproducible numbers.

### 0a. Choose benchmark shapes

Create a CSV with columns `M,N,K,MT_M,MT_N,MT_K` containing shapes that span the target workload.
Include large shapes (saturate the GPU), medium shapes (representative), and small shapes (launch-overhead canaries).
Large and medium shapes are the optimization targets; small shapes are regression canaries.

### 0b. Run the benchmark

Use the appropriate benchmark script for your kernel type.
For MXFP4 GEMM kernels, the benchmark is `wave_lang/kernel/wave/perf/benchmark_mxfp4.py`.
For other kernel types, write a standalone benchmark that uses `rocprofv3 --kernel-trace --stats` for timing.

```bash
cd <worktree>
WAVE_CACHE_ON=0 PYTHONPATH=$PWD \
python -u wave_lang/kernel/wave/perf/benchmark_mxfp4.py \
  --shapes <shapes.csv> \
  -o baseline.csv \
  --backend asm \
  --warmup-iters 3 --benchmark-iters 10
```

Record the TFLOPS and runtime (us) for every shape.
Compute % of theoretical peak for the target hardware and data type.
This is the baseline you compare all changes against.

### 0c. Run multiple times

GPU benchmarks have variance (20-40 TFLOPS on large shapes).
Run at least 2-3 times to establish confidence intervals.
Only trust improvements that exceed the noise floor.

## Phase 1: Static analysis (no GPU required)

### 1a. Dump and inspect AMDGCN assembly

For the waveasm backend, get the assembly output by setting `compile_to_asm=True` in `WaveCompileOptions`, or by using the `--dump-dir` flag with the benchmark script to save compilation artifacts.

Alternatively, compile programmatically and print the assembly:

```python
gemm = get_compiled_kernel(shape, block, backend="asm")
print(gemm.asm)
```

### 1b. Key metrics to extract from assembly

| Metric | Where to find | Implication |
|--------|---------------|-------------|
| VGPR count | `.amdhsa_next_free_vgpr` | Determines occupancy: floor(512 / vgprs) on CDNA4.  >256 means occupancy 1. |
| SGPR count | `.amdhsa_next_free_sgpr` | Usually not the bottleneck; >106 on gfx9 is a problem. |
| LDS size | `.amdhsa_group_segment_fixed_size` | Must fit in CU LDS budget (64-128 KB depending on arch), shared across waves in the workgroup. |
| MFMA count per loop | Count `v_mfma_*` instructions | Should match expected: M_tiles * N_tiles * K_unroll. |
| `v_mov_b32` per loop | Count in loop body | Register copy overhead from waveasm's loop-carried value handling. |
| `buffer_load_*` per loop | Count global loads | Data + scale loads per iteration. |
| `ds_read_*` per loop | Count LDS reads | Shared-memory reads per iteration. |
| `s_barrier` per loop | Count barriers | Fewer is better for asymmetric schedules. |
| `s_waitcnt` per loop | Count wait instructions | Check vmcnt/lgkmcnt thresholds for stall severity. |
| `buffer_load_ubyte` | Grep for ubyte loads | Unmergeable scale loads indicate a tile-size alignment issue. |
| `s_nop` count | Count NOPs | Hazard mitigation overhead. |
| Total instructions in loop | Count between loop label and branch | Overall loop body cost. |

### 1c. Compute MFMA utilization from assembly (static estimate)

A static estimate of MFMA efficiency is:

```
utilization = (n_mfma * mfma_issue_cycles) / total_loop_instructions
```

where `mfma_issue_cycles` comes from the ISA reference (see `benchmark/wave_att_analyzer.py` for a table of known values).
For example, `v_mfma_scale_f32_16x16x128_f8f6f4` takes 16 cycles on gfx950.
If the loop has 80 MFMAs and 243 total instructions, the static utilization is `80*16/243 = 52.7%` -- but this ignores instruction-level parallelism, so actual utilization from tracing is the ground truth.

### 1d. Identify the bottleneck category

- **Occupancy-limited**: VGPR count forces low occupancy (1-2 waves/SIMD), so latency hiding is poor.
- **Compute-bound**: MFMA utilization is high but there are gaps between MFMAs (visible as bubbles in ATT trace).
- **Memory-bound**: Frequent `s_waitcnt vmcnt(N)` stalls; memory latency is not hidden by compute.
- **Barrier-bound**: Too many barriers per loop, or barriers placed so that waves idle waiting for synchronization.
- **Instruction overhead**: High `v_mov_b32` count or excessive scalar ALU for address computation, G2S SRD setup, etc.

### 1e. Compare against a known-good configuration

If another block size achieves better performance, dump its assembly too and compare instruction counts.
Differences reveal block-size-specific inefficiencies.

## Phase 2: Kernel tracing with rocprofv3

### 2a. Basic kernel trace

```bash
source <repo>/rocprof-trace-env
rocprofv3 --kernel-trace --stats -d /tmp/trace_output -- \
  python -u <benchmark_or_trace_script.py>
```

This gives per-dispatch kernel duration and basic stats.

### 2b. Hardware performance counters

```bash
rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VMEM,SQ_INSTS_LDS,SQ_INSTS_VALU,SQ_WAIT_INST_VMEM,SQ_WAIT_INST_LDS \
  -d /tmp/pmc_output -- python <trace_script.py>
```

Key counters:

- `SQ_INSTS_VALU / VMEM / LDS`: instruction mix
- `SQ_WAIT_INST_VMEM / LDS`: stall cycles (memory vs LDS bound)
- `SQ_WAVES`: total wave launches
- `FETCH_SIZE / WRITE_SIZE` (L2): actual memory bandwidth utilization

### 2c. Advanced Thread Trace (ATT) -- cycle-level profiling

ATT is the most powerful tool for finding instruction-level bottlenecks.

**Write a trace script** that compiles the kernel, warms up, then dispatches once:

```python
import os, torch
os.environ["WAVE_CACHE_ON"] = "0"

gemm = compile_kernel(shape, block, backend="asm")
inputs = prepare_inputs(shape, block)

for _ in range(3):
    gemm(*inputs)
torch.cuda.synchronize()

gemm(*inputs)
torch.cuda.synchronize()
```

**Write an `att.json` filter** to capture only the kernel of interest:

```json
{
    "jobs": [{
        "kernel_include_regex": "<kernel_name>",
        "kernel_exclude_regex": "",
        "kernel_iteration_range": "[1]",
        "advanced_thread_trace": true,
        "att_parse": "trace",
        "att_target_cu": 0,
        "att_shader_engine_mask": "0xF",
        "att_simd_select": "0xF",
        "att_buffer_size": "0x60000000"
    }]
}
```

**Capture the trace:**

```bash
rocprofv3 --att \
  --att-library-path <path-to>/rocprof-trace-decoder/opt/rocm/lib/ \
  -i att.json \
  -d /tmp/att_output -- \
  python trace_script.py
```

### 2d. Compute MFMA utilization from ATT trace

Use `benchmark/wave_att_analyzer.py` to compute per-wave and per-SIMD MFMA utilization:

```bash
python benchmark/wave_att_analyzer.py /tmp/att_output --inst mfma -v
```

This computes:

```
efficiency = (n_mfma_per_loop * mfma_issue_cycles) / single_loop_actual_cycles * 100
```

The output includes per-wave breakdowns and per-SIMD utilization (accounting for co-executing waves sharing the MFMA unit).
This is the ground-truth XDL/MFMA utilization metric -- the single most important number for optimization.

### 2e. Analyze ATT output

- `stats_ui_output_*.csv`: per-instruction cycle counts
- `ui_output_*` directories: visualizable with ROCprof Compute Viewer
- Look for: bubbles between MFMAs, long vmcnt/lgkmcnt waits, barrier serialization, register dependency stalls

To visualize, tar the output directory and open with ROCprof Compute Viewer (requires GUI; transfer to a machine with a display if needed).

## Phase 3: Schedule parameter tuning

The schedule is the highest-leverage optimization knob in Wave.
Schedule files live in `wave_lang/kernel/wave/schedules/`.
Templates live in `wave_lang/kernel/wave/templates/`.

Read `docs/wave/wave_schedule.rst` for the scheduling API (pipeline stages, tag-based node selection, interleave operations).

### 3a. Interleave pattern tuning

The `interleave_operations` function places memory ops between MFMAs.
Its parameters:

```python
base_offsets = [0, 3, 2, 0]   # starting position for each op group
base_intervals = [4, 4, 2, 4] # how often each op group is injected
```

These control how frequently memory operations are interleaved with compute.
Tuning strategy:

- Fewer MFMAs per partition may benefit from tighter intervals (e.g. `[3,3,2,3]` instead of `[4,4,2,4]`).
- Tighter intervals inject memory ops more often, better matching the compute/memory ratio for smaller tile configurations.
- Too-tight intervals (e.g. `[2,2,1,2]`) can cause validation failures -- always check numerical correctness.
- Asymmetric intervals (e.g. `[2,3,2,3]`) tend to regress performance.

**Recommended approach**: create 3-5 variants with different offset/interval combinations, benchmark all on 3 representative shapes, pick the winner.

### 3b. MemoryCounterWaitBarrier (waitcnt) tuning

The `MemoryCounterWaitBarrier` controls when waves synchronize on memory operations.
Relaxing the threshold allows more overlap between memory and compute:

```python
MemoryCounterWaitBarrier(load=expected_count, lds=0)       # default: exact
MemoryCounterWaitBarrier(load=expected_count + 2, lds=0)   # relaxed: +2 slack
```

- `+2` slack is often the sweet spot: lets hardware overlap more memory with compute while maintaining correctness.
- `+3` and beyond frequently cause validation failures.
- `load=0` (full drain) dramatically hurts performance.

**Always validate correctness** after changing waitcnt thresholds.

### 3c. What to try, in priority order

| Optimization | Effort | Potential impact | Risk |
|---|---|---|---|
| Interleave interval tuning | Low | 1-5% | Low (validate) |
| Waitcnt slack (+2) | Low | 1-4% | Medium (may break correctness) |
| Combined interleave + waitcnt | Low | 1-5% (composes) | Medium |
| Alternative wave shapes | Medium | Variable | Medium |
| Reduce prefetch depth | Medium | May reduce VGPRs | High (correctness) |
| New schedule structure (pingpong) | High | Variable | High |
| VGPR reduction for occupancy gain | High | Up to 2x | Very high |
| Waveasm backend changes | Very high | 5-15% (v_mov elimination) | Very high |

## Phase 4: Experiment execution

### 4a. Parallel experiments via subagents

For independent optimizations, launch parallel subagents each working in their own worktree and branch.
Use a consistent branch naming scheme supplied by the user.

### 4b. Each experiment must

1. Modify exactly one schedule parameter (or a small, coherent set).
2. Validate numerical correctness on at least one shape.
3. Benchmark on the same shape set as the baseline.
4. Record results in a structured markdown table with TFLOPS and delta vs baseline.

### 4c. Compare results

Collect all experiment results into a single comparison table.
Look for:

- Improvements that are **consistent** across multiple shapes.
- Improvements that exceed the noise floor (typically >1% relative).
- **No regressions** on any shape.
- Combinations that compose well (e.g. interleave tuning + waitcnt relaxation).

## Phase 5: Register pressure analysis (advanced)

If occupancy is the bottleneck (VGPRs > 256 on CDNA4), analyze register usage.

### VGPR budget breakdown

- **Accumulator VGPRs**: M_tiles * N_tiles * 4 (for f32 output); irreducible for a given tile config.
- **Data VGPRs**: For global-to-VGPR loads with double buffering, each matrix operand loaded from global memory uses tiles * loads_per_tile * 4 * 2 (current + prefetched) VGPRs.
- **LDS-loaded data VGPRs**: partially reusable across partitions.
- **Scale VGPRs**: relatively small.
- **Address computation VGPRs**: ~10-20.
- **v_mov overhead**: waveasm uses extra VGPRs for loop-carried value copies (WAR hazard avoidance and LDS double-buffer ping-pong swaps).

### Reduction strategies

- Reduce prefetch depth (single-buffer instead of double-buffer), but this hurts latency hiding.
- Use a different tile configuration with fewer MFMAs (smaller accumulator footprint).
- Load operands just-in-time rather than pre-loading all tiles.
- Improve waveasm register coalescing (backend change -- high effort).

## Key environment variables

| Variable | Purpose |
|---|---|
| `WAVE_CACHE_ON=0` | Disable kernel caching (required for testing changes) |
| `PYTHONPATH=<worktree>` | Ensure the correct Wave code is used |
| `WAVE_DEBUG_LIVENESS=1` | Show liveness analysis results |
| `WAVE_DEBUG_REGALLOC=1` | Show register allocation decisions |
| `WAVE_LDS_DSREAD_OFFSET_DEBUG=1` | Show ds_read offset optimization |
| `WAVE_STRICT_FORMATTER=1` | Strict instruction formatting validation |

## Quick debugging tips

- If a schedule change causes validation failure, it likely broke memory ordering.
  Revert the change and try a smaller perturbation.
- If assembly compilation fails with "register exhaustion", the schedule change increased VGPR pressure beyond 512.
  Check if new interleave patterns cause more values to be live simultaneously.
- Use `--dump-dir` with the benchmark to save MLIR/assembly for inspection without a separate compilation step.
- GPU memory faults on stores usually mean OOB writes from partial workgroups.
  Check SRD NUM_RECORDS for the output buffer.
- If you see `buffer_load_ubyte` where you expect `buffer_load_dword`, the scale merge pass failed.
  Check if the tile size is divisible by 32.

TODO - some of these failures are specific to specific tasks (eg. mxfp4 with scale read merging).  They should probably be details introduced in context for those specific tasks.
