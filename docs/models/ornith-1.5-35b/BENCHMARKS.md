# Ornith-1.5-35B benchmarks

Scope: native Windows port, single AMD Strix Halo (gfx1151, Ryzen AI Max+ 395,
110 GiB visible carve-out), build under `feature/window-native` with the
TheRock HIP 7.15 nightly runtime, 2026-09-29. `gufo bench` depth 0, C=1.
The Linux production numbers land with the Linux qualification run.

## gufo bench (internal path)

| Test  | Workload | Throughput (tok/s) |
| --- | --- | --- |
| pp2048 | depth 0, 2048-token prompt | 2926.5 ± 4.3 |
| tg128 | autoregressive decode | 63.4 ± 0.3 |
| tg128 + MTP | `--speculative mtp`, 7-draft cap | 79.0 ± 0.4 |

Decode rows are the native-decode build (2026-09-29 perf pass): dense
decode-width projections read the artifact's own Q4_K bytes through the
routed vector kernel (E=1) instead of a dequantized Q8_0 copy, +11% AR.
The first MTP measurement (89.6 tok/s, 79% acceptance) ran on a
hyper-repetitive greedy stream; after the same pass the stream is natural
text and the like-for-like speedup over AR is +25% at 63/88 drafts (72%).
Prefill is unchanged (wide-batch Q8_0 views).

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
