# Ornith-1.5-35B benchmarks

Scope: native Windows port, single AMD Strix Halo (gfx1151, Ryzen AI Max+ 395,
110 GiB visible carve-out), build under `feature/window-native` with the
TheRock HIP 7.15 nightly runtime, 2026-09-29. `gufo bench` depth 0, C=1.
The Linux production numbers land with the Linux qualification run.

## gufo bench (internal path)

| Test  | Workload | Throughput (tok/s) |
| --- | --- | --- |
| pp2048 | depth 0, 2048-token prompt | 2957.7 ± 1.8 |
| tg128 | autoregressive decode | 57.3 ± 0.0 |
| tg128 + MTP | `--speculative mtp`, 7-draft cap | 89.6 ± 0.8 |

MTP acceptance on the deterministic bench text: 83/105 drafts (79%).

## HTTP serving (`gufo serve llm -c 4096`, server-reported timings)

| Mode | Prefill (tok/s) | Decode (tok/s) | Notes |
| --- | --- | --- | --- |
| AR | 2379.1 ± 62.6 | 49.0 ± 0.3 | ~1.2k-token prompts, n=128, ttft ≈ 540 ms |
| MTP | — | 55.5–61.4 | 63–79% draft acceptance (fiction vs formulaic text) |

HTTP runs ~2% below the internal path on prefill at 1.2k tokens (chunked
prefill at 2048-token capacity; the native run measures full 2048-token
chunks) and ~15% on AR decode (per-request tokenize/template/sampling
overhead plus the shorter steady-state window). Draft-acceptance counters
populate per request (`usage.draft_tokens*`).

## Memory

Weights resident ≈ 21.2 GiB device (20.2 GiB artifact plus dequantized-Q8_0
dense copies and the Q6_K→Q8_0 head); peak device use 24.7 GiB at 4096
context with one session — far inside the carve-out, nothing spills to
WDDM shared memory.

## TODO

- Linux gfx1151 qualification numbers (AGENTS.md matched-hardware rule).
- Deeper-context decode sweep (8k/32k) once the Windows port retains its
  long-context gates.
