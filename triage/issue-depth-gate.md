Follow-up from #6 (fix landed in `3264da5`+`4b4d7c1`, re-validated on the v0.9.0 base; full write-up in `triage/TRIAGE.md` on branch `triage/issue-6-empty-turns`).

## Finding

The per-row PLE / LM-head fix restored width-1 parity and made speculative verify behave like autoregressive decoding — but only inside the **dense-attention region**. At context depth ≥ `indexer_top_k` (2048 tokens, `compress_ratio` 4), the width-2 verify frontier diverges from what width-1 decoding computes for the same token:

- **Positions < 2048:** prefill exact, width-1 identical for 250+ steps, post-reject rollback restore bit-exact; streams flip only on near-ties (first at emitted token ~40, e.g. 跟进/跟) — the documented residual.
- **Positions ≥ 2048:** the width-2 verify frontier differs from width-1 by **max|Δ|≈10 across all 248,320 logits from the first verify cycle**, and the post-reject restore test diverges (max|Δ|≈0.52). This reproduces identically on the pre-merge fix head (`4b4d7c10`) and on v0.9.0 (`c1028db1`), in both incremental and from-scratch build directories — it is **not** a regression from the v0.9.0 merge; every build measured behaves the same.

## Probable mechanism

The sparse attention/indexer decode path (positions ≥ `indexer_top_k`) is row-count-dependent at n≥2: verify rows run through batched reductions whose numerics differ from the single-token path. The per-row PLE/head fix never claimed that path; MoE grouped GEMM and GDN likely contribute on top.

## Deterministic reproducer

```text
qwen38_flash_next_parity_probe --model <UD-Q4_K_XL first shard> \
  --mtp-model <mtp Q8_0> --rounds 6
```

The default 6-round prompt is 7,407 tokens (entirely in the sparse region) and shows `phase 2 (cycle-2): step 0 … frontier max|d|≈10 differing=248320` plus `restore-test step 0: DIVERGES`. With `--rounds 1` (1,423 tokens, dense region) the same binary passes with only the documented token-40 near-tie flip. Note: every validation of #6's fix used the 1-round prompt; the 6-round protocol is what exposes this.

## Impact

At the fixture's working depths (2.4–4K tokens) every speculative cycle re-rolls near-ties, so MTP's temp-1.0 empty-turn risk stays several times the AR-only rate: latest arm 9/56 (16.1%), pooled 15/126 (11.9%) vs AR-only 1/56 (1.8%). Mitigation until this lands: `--speculative off` (0/56 across two full runs).

## Acceptance bar

- `--rounds 6` probe: phase-2 frontier max|Δ|=0 from cycle 1 and an IDENTICAL restore test.
- No tg/pp regression; empty-turn exam arm at/below AR-adjacent rates.

## Scope

Row-count-invariant decode kernels for the sparse attention/indexer path (then audit MoE grouped GEMM and GDN), or per-row routes where the perf cost is acceptable.
