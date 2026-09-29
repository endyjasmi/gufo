# Ornith-1.5-35B experiments

- 2026-09-29: ported the qwen35moe runtime from the Flash-Next package by
  stripping the PLE n-gram, hyper-connection and sparse-indexer stages;
  attention runs dense only. Retained.
- Reusing the vendored qfn_mmq kernel build instead of duplicating it:
  retained (generic quant kernels; ds4 precedent).
- Native Q4_K routed down projections on the WMMA tier (48-row tiles):
  retained; Q6_K downs dequantize to Q8_0 at load.
- 2026-09-29 decode pass: dense decode-width projections keep a raw Q4_K
  byte view beside the Q8_0 wide-batch view and run `qfn_mmq_moe_vec` E=1
  (shared expert via `moe_gated_vec`); +11% AR (57.3 to 63.4 tok/s).
  Retained. The MTP draft's split fc_embedding/fc_hidden projections must
  NOT carry the combined tensor's whole-row view — the first cut leaked it
  and the draft proposals collapsed to 0% acceptance. Rejected: event-poll
  sync and pipeline-replay probes (WDDM overhead is not on the critical
  path; graph replay confirmed the decode graph itself is GPU-bound at
  ~17 ms/token, i.e. bandwidth).
- 2026-09-29 benchmark pass (native vs HTTP, concurrency): prepared-cohort
  HTTP decode lands within 2% of native AR (62.1 vs 63.2 tok/s) and scales
  to 198.3 tok/s aggregate at C8 with hash-exact completions. Retained
  (sequential session preparation during cohort setup is required; see
  QUALITY.md). Two engine findings, unfixed, both reproducible:
  (1) concurrent prefill requests interleave scheduler chunks and the
  width-sensitive prefill routes (F32 router hipBLAS sgemv; F16-route
  threshold at 2048 rows) shift logits by ulps, so a session prefilled
  alongside another can decode to different greedy text than the isolated
  reference — fix direction: width-invariant router kernel or chunk-invariant
  routes; (2) co-batched speculative verification beyond 8 total rows leaves
  the q8_1 vector kernels for the tiled fallback with different quantization
  and reduction, tripping the greedy-equivalence gate at C4 (mixed) / C6
  (repetitive) — fix direction: cap co-batched verify rows at the vector
  contract or make the tiled path reproduce it.
- 2026-09-29 Q8_0 artifact pass: every projection keeps its native Q8_0
  bytes end to end. Three engine changes, retained: (1) `DequantizeRow`
  decodes Q8_0 sources — the raw-block memcpy fallback would have
  corrupted the dense stacks and the alpha/beta stacks of a Q8_0 artifact;
  (2) `CopyDequantQ8_0` streams Q8_0 parts straight to the device instead
  of a dequant+requant round trip, and the split MTP projection drains the
  staging pipeline before it frees the combined buffer — that free was
  only safe while the upload was synchronous (first Q8_0 load died in the
  staging worker with `invalid argument`); (3) routed Q8_0 gate/up joins
  the F16 WMMA tier (the routed kernel already decoded Q8_0 downs):
  pp2048 2530 → 3074 tok/s, +21% over the int8 raw-moe fallback and +7%
  over the Q4_K_M build. Findings: single-stream AR decode is
  bandwidth-bound and pays the artifact's 1.7× expert bytes (-17% vs
  Q4_K_M), converging to Q4_K_M at C4-C8 where batched GEMVs are
  MAC-bound (C6 +2%); HTTP MTP loses (-14/-23%) even though native MTP
  gains (+3%), because acceptance drops (55/58% vs 63/78%) while the
  adaptive policy drafts more per cycle — sharper draft and target logits
  agree less often on near-ties. llama.cpp b11243's Q8_0 greedy is NOT
  ground truth for this artifact: it diverges from our F32 oracle at a
  0.6-logit near-tie where the GPU matches the oracle; use the oracle
  gates in QUALITY.md for Q8_0 comparisons.
