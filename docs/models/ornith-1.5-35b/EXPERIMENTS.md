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
