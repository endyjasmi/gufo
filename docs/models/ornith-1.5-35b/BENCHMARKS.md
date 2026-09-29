# Ornith-1.5-35B benchmarks

Scope: native Windows port, single AMD Strix Halo (gfx1151, Ryzen AI Max+ 395,
110 GiB visible carve-out), build under `feature/window-native` with the
TheRock HIP 7.15 nightly runtime, 2026-09-29. `gufo bench` depth 0, C=1, and
`gufo serve llm` HTTP (chat completions, greedy, temperature 0, thinking off).
Greedy HTTP completions are hash-compared against the isolated AR reference;
every published row matches. The llama.cpp reference columns are TODO on this
host (the pinned reference runtime is Nix/Linux only; the Linux qualification
run fills them).

[Quality and measurement details](QUALITY.md) · [Experiments](EXPERIMENTS.md)

## Native vs HTTP (single user, depth 0)

Native is `gufo bench` (kernel-iteration path, direct engine calls); HTTP is
the serving path with a prepared 2040-token prompt (one token generated to
anchor the cache, then the measured request reuses the whole prefix).
Prefill is autoregressive in every mode; MTP changes decode only.

| Mode | Native pp2048 (tok/s) | HTTP pp (tok/s) | Native tg128 (tok/s) | HTTP tg128 (tok/s) | HTTP gain vs native |
| --- | ---: | ---: | ---: | ---: | ---: |
| Autoregressive | 2870.9 ± 14.7 | 2653.5 | 63.21 ± 0.09 | 62.11 | -1.7% |
| MTP, mixed text | 2887.4 ± 5.0 | 2618.8 | 78.90 ± 0.18 | 73.18 | -7.3% |
| MTP, repetitive text | — | 2618.3 | 78.90 ± 0.18 | 79.50 | +0.8% |

HTTP decode sits within 2% of native for AR; the MTP gap on mixed text is
host-side draft chaining (a full-vocabulary head pass plus a stream
synchronization per draft step, which the native bench also pays but the HTTP
request path amortizes worse). Draft acceptance: 63% mixed, 78% repetitive.
Earlier (2026-09-29, unprepared ~1.2k-token prompts) HTTP rows of pp 2379 /
tg 49.0 measured a shorter steady-state window with full prefills; the
prepared-cohort method above replaces them.

## Single user, autoregressive

Depth 0 only on this host; deeper cached-prefix sweeps belong to the Linux
qualification run.

<!-- bench:single-ar -->
| Ornith Q4 AR<br>Depth (tokens) | Gufo pp (tok/s) | llama.cpp pp (tok/s) | Gain | Gufo tg (tok/s) | llama.cpp tg (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2653.48 | TODO | TODO | 62.11 | TODO | TODO |
<!-- /bench -->

## Single user, MTP

<!-- bench:single-mtp -->
| Ornith Q4 MTP<br>Depth (tokens) | Gufo pp (tok/s) | Gufo tg mixed (tok/s) | llama.cpp AR tg mixed (tok/s) | Gain mixed | Gufo tg repetitive (tok/s) | llama.cpp AR tg repetitive (tok/s) | Gain repetitive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2618.81 | 73.18 | TODO | TODO | 79.50 | TODO | TODO |
<!-- /bench -->

## Multiple users, autoregressive

Same pp2048 prose prompt as single-user d0, tg128, context 4096 per user.
Every session is prepared (full prefill plus one anchored token, one session
at a time) before the measured cohort is released together; throughput sums
individual request decode rates. Every level's completions are hash-identical
to the isolated AR reference.

<!-- bench:multi-ar -->
| Ornith Q4 AR<br>Users | Gufo AR (tok/s) | llama.cpp AR (tok/s) | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 61.67 | TODO | TODO |
| 2 | 94.48 | TODO | TODO |
| 4 | 150.36 | TODO | TODO |
| 6 | 172.35 | TODO | TODO |
| 8 | 198.26 | TODO | TODO |
<!-- /bench -->

Decode scales to 3.2x C1 at C8; the aggregate stops growing linearly because
the quantized GEMV kernels become multiply-accumulate-bound once several rows
share each weight read.

## Multiple users, MTP

Same prompts and cohort method as the AR table. C1/C2 complete and match the
AR reference exactly. C4 (mixed) and C6 (repetitive) fail the greedy-equivalence
gate: co-batched speculative verification crosses the 8-row vector-kernel
contract, whose tiled fallback is not bit-compatible, so wider levels are
withheld (see `artifacts/unavailable.json`). The engine fix is tracked in
[EXPERIMENTS.md](EXPERIMENTS.md).

<!-- bench:multi-mtp -->
| Ornith Q4 MTP<br>Users | Gufo mixed (tok/s) | llama.cpp mixed (tok/s) | Gain | Gufo repetitive (tok/s) | llama.cpp repetitive (tok/s) | Gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 72.61 | TODO | TODO | 78.69 | TODO | TODO |
| 2 | 105.23 | TODO | TODO | 116.90 | TODO | TODO |
| 4 | TODO | N/A | N/A | 172.00 | TODO | TODO |
| 6 | TODO | N/A | N/A | TODO | N/A | N/A |
| 8 | TODO | N/A | N/A | TODO | N/A | N/A |
<!-- /bench -->

## Memory

Weights resident ≈ 21.2 GiB device (20.2 GiB artifact plus dequantized-Q8_0
dense copies and the Q6_K→Q8_0 head); peak device use 24.7 GiB at 4096 context
with one session and 26.1 GiB with `--sessions 8` — far inside the carve-out,
nothing spills to WDDM shared memory.

## TODO

- llama.cpp reference columns (AR and MTP, single-user and concurrency):
  Nix-pinned reference runtime; land with the Linux gfx1151 qualification run.
- Depth sweeps (4K-32K cached prefixes) for AR and MTP, loading time, and the
  deep-context memory rows: same.
- Engine work before wider MTP concurrency can be published: make batched
  speculative verification reproduce the single-session arithmetic (and, for
  serving correctness in general, make concurrent prefill chunk interleaving
  width-invariant — see EXPERIMENTS.md).
