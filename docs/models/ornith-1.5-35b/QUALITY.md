# Ornith-1.5-35B quality

## Validation gates (2026-09-29, Windows port build)

- Scalar oracle: the float32 reference (`gufo_qwen35moe_reference`) is
  token-identical with llama.cpp greedy completion on the same raw prompt
  ("The capital of France is" → " Paris.", llama.cpp b11243 CPU build).
  This pins the graph semantics: pre-norm residual + post-attention norm,
  Gated DeltaNet with SiLU output gate and tiled value-head pairing
  (v-head h reads key head h % 16), gated GQA with sigmoid output gate,
  softmax top-8 renormalized MoE with gated shared expert.
- GPU vs oracle: per-position logits agree at cosine 0.988–0.996 with
  matching argmax over mixed prompt positions (residual gap = the GPU's
  Q8_0-requantized dense weights, F16 activations/KV, and WMMA attention
  against exact F32 on the original Q4_K).
- Batched-prefill gate: `gufo bench --validate-prefill 64` passes
  (scalar_winner_rank=1, cosine 0.9977, finite).
- Native decode views (2026-09-29 perf pass): GPU vs oracle cosine
  0.979-0.996 across two prompts; 6/8 argmax positions agree exactly and
  the two disagreements are near-ties (both paths quantize activations;
  the native path keeps the artifact's own Q4_K weights). Greedy streams
  are deterministic under repetition. MTP draft acceptance restored at
  63/88 (72%) after fixing the split-projection view leak.
- Operator tests: `qwen35moe.attention_ops` (16-head WMMA vs per-token
  reference, chunk-invariance exact) and `qwen35moe.gdn_ops` (32 value
  heads, conv + row-split contracts) pass on gfx1151.

## Known gaps

- Serve-path draft-acceptance counters were verified manually (63–79%);
  the automated sweep for long-context MTP-after-restore parity from the
  Flash-Next port applies here too.
- The MTP cost priors (`mtp_costs.hpp`) still hold the Flash-Next
  measurements; the draft-length controller converges online, but the
  priors should be re-measured for this model. Do not retune them before
  the per-draft cost drops: at the measured ~5.9 ms/draft (full head pass
  plus a stream synchronization per draft), the current priors already
  steer to the throughput-optimal short chains (forced 7-draft cycles
  measure 47.9 tok/s against 79 adaptive).

## Benchmark method (2026-09-29, Windows host)

- Driver: `tools/bench/model-bench.py` with
  `docs/models/ornith-1.5-35b/artifacts/bench.json` (Gufo-only; no llama.cpp
  reference exists on this host — reference columns are TODO until the Linux
  qualification run). Server: `gufo serve llm --think off
  --max-pending-per-client 8 --sessions C`, greedy, seed-free temperature 0.
- Native numbers: `gufo bench -p 2048 -n 128 -r 3`, AR and
  `--speculative mtp --min-draft-tokens 1 --draft-tokens 7`.
- Concurrency cohorts prepare every session with the exact 2040-token prose
  prompt plus one anchored token, one session at a time, then release the
  cohort together (`prepared_prefill_sequential`). Decoding one session at a
  time during preparation is required for exactness: two concurrent 2040-token
  prefills interleave scheduler chunks, and the width-sensitive prefill routes
  (the F32 router's hipBLAS sgemv among them) shift logits by ulps, flipping
  rare near-tie expert selections; the resulting session state decodes to
  different greedy text than the isolated reference. Sequentially prepared
  cohorts are hash-identical at every AR width (C1-C8) and for speculative
  verification up to the 8-row vector-kernel contract.
- Withheld rows: batched speculative verification co-batching sessions beyond
  8 total rows leaves the vector kernels for the tiled fallback, which does
  not reproduce the vector path's arithmetic; mixed text trips the gate at C4,
  repetitive text (wider logit margins) at C6. See `artifacts/unavailable.json`.
- Completion hashes: every published Gufo row (single-user AR, single-user
  MTP both workloads, multi-user AR C1-C8, multi-user MTP C1/C2 mixed and
  C1-C4 repetitive) matches the isolated AR C1 reference hash exactly.
- Reproduction: `python tools/bench/model-bench.py --model ornith-1.5-35b
  --gufo <binary> --gguf <artifact> run --target gufo --table single-ar,
  single-mtp,multi-ar,multi-mtp`, then `render --no-charts`.
  Windows needs the `servers.py` terminate fallback (no process groups).
