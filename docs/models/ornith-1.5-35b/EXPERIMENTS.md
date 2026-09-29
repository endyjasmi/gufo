# Ornith-1.5-35B experiments

- 2026-09-29: ported the qwen35moe runtime from the Flash-Next package by
  stripping the PLE n-gram, hyper-connection and sparse-indexer stages;
  attention runs dense only. Retained.
- Reusing the vendored qfn_mmq kernel build instead of duplicating it:
  retained (generic quant kernels; ds4 precedent).
- Native Q4_K routed down projections on the WMMA tier (48-row tiles):
  retained; Q6_K downs dequantize to Q8_0 at load.
