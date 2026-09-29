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
  priors should be re-measured for this model.
