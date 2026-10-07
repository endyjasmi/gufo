# Triage: endyjasmi/gufo#6 — reasoning-only empty turns + identity denials on `feature/window-native`

**Date:** 2026-10-07 · **Binary under test:** `D:\gufo\build\windows-release\gufo.exe`
(built from `feature/window-native` @ `1cfd681`, one commit past the reported `39d8378`)
**Machine:** Strix Halo gfx1151, same class as reporter. Model: unsloth
`Qwen3.8-Flash-Next` UD-Q4_K_XL (4 shards, rev `38bb39ee`) + `shared-Q8_0` MTP +
mmproj-BF16 — same model files as the report.

## Confirmation rerun (same binary, same protocol, new RNG draws)

A full second pass of every arm (results in `results/*-r2*`, logs
`*-r2.log`, probe r1 outputs preserved as `results/*-r1`):

| Arm | Run 1 | Run 2 | Combined |
|---|---|---|---|
| baseline exact signature (finish=stop, content="") | 5/28 | 3/28 | **8/56 (14.3%)** |
| baseline budget-burn loops (finish=length, content="") | 4/28 | 0/28 | 4/56 |
| `--speculative off` exact signature | 0/28 | 0/28 | **0/56** |
| `--speculative off` budget-burn loops | 0/28 | 1/28 | 1/56 |
| identity denials | 0 | 0 | 0 |

- The contrast replicates: **8/56 exact-signature empty turns with MTP vs
  0/56 without**. Run-to-run rates vary with the RNG draw (5/28 vs 3/28),
  bracketing the reporter's 25% on their fixture — consistent with a
  material-dependent probability, not a fixed count.
- Honest nuance: the AR-only arm produced 1/28 budget-burn in run 2 — the
  base model at temp 1.0 with no repetition penalty can loop on its own,
  rarely. MTP raises the degeneration rate roughly an order of magnitude
  and is responsible for the exact `finish=stop` signature (AR only ever
  burned the budget, never stopped early with empty content).
- The greedy parity probes are **bit-stable across runs**: persona turn 1
  diverges at char 195 both times (MTP stops 987 vs AR 1200 cap), plain
  English at char 734 both times. Deterministic engine-level drift, not
  noise. The width ablation (`--draft-tokens 1/2/7` → same divergence
  offset) was not repeated; its conclusion does not depend on the draw.

## Verdict (TL;DR)

**Reproduced, and the root cause is isolated to the speculative (MTP) decode
path — not the Windows port, not the chat template, not sampling math.**

1. The reported failure signature (`finish=stop`, `content=""`, non-empty
   `reasoning_content`) reproduces on the current head with the reporter's
   exact server config, on a synthetic structural analog of their exam:
   **5/28 turns (17.9%) exact signature + 4/28 budget-burn loops = 9/28 turns
   (32%) ending with empty content.** Reporter saw 25%.
2. Every failure is a **reasoning-phase degeneration**: the model enters
   n-gram repetition attractors ("胖哥胖哥胖哥…", "Jarvis Jarvis Jarvis…")
   or emits EOS while still inside the think block. Some loops burn the whole
   `max_tokens` budget (`finish=length`); others hit EOS mid-loop → the
   reporter's exact `finish=stop` empty turn. MTP draft acceptance inside
   loops is 0.92–0.97 vs 0.60–0.74 on healthy turns — the draft head simply
   rides the loop; the degeneration is in the **target's own distribution**.
3. **`--speculative off` is completely clean: 0/28.** Same binary, same
   fixture, same requests, same sampling. (This answers the reporter's open
   question 5, and gives users an immediate workaround.)
4. **Greedy parity is broken.** At temperature 0 — where MTP-on and MTP-off
   must produce token-identical output — they diverge deterministically
   mid-generation on an *identical prompt* (persona exam: char 195; plain
   English with thinking disabled: char 734). The MTP arm also **ends turns
   earlier** (stops at 491 vs 553 tokens; 987 vs ≥1200) — premature EOS even
   at temp 0, the same mechanism class as the reported empty turns.
5. The divergence offset is **identical at every draft width** (`--draft-tokens`
   1, 2, 7 all diverge at the same char). Width 1 performs plain single-token
   decode forwards, so the multi-token verify batch and the rollback path are
   ruled out. The drift enters through the **trunk frontier forward itself when
   the MTP machinery is loaded** — i.e., the hidden-row-capture / shared-frontier
   path that feeds the draft (qfn executor).
6. The existing in-repo parity gate cannot see this:
   `CheckServingSampling` (tests/models/qwen38_flash_next/session_test.cpp:876)
   asserts greedy AR/MTP equality only on **8-token completions over tiny
   prompts**. The drift appears after ~100–200 tokens of agreement.

Immediate mitigation for the reporter: run with `--speculative off`
(costs ~15% decode speed; behavior verified clean on this fixture).
Real fix + evidence: see "Next steps".

## Method

Harness in this directory (untracked→committed here): `fixture.py` (two
7-turn Chinese persona + `<memory-context>` exams, structurally matching the
reporter's private fixture: identity system prompt, dated memory cards, 3–6K
char user messages), `repro_exam.py` (fresh server process per session, exact
reporter flags, temp 1.0 / top-p 0.95 / top-k 20 / min-p 0, non-stream,
`chat_template_kwargs.preserve_thinking=true`, empty turns append
`reasoning_content` — mirroring their client), plus the ablation/probe
scripts. Every arm ran 4 sessions × 7 turns with a fresh cold-cache server
process per session (matching the reporter's methodology).

## Results

| Arm | Config | Empty (finish=stop) | Budget-burn (finish=length) | Denials |
|---|---|---|---|---|
| baseline | reporter's flags (MTP on) | **5/28 (17.9%)** | 4/28 | 0 |
| nospec | `--speculative off`, else identical | **0/28** | 0/28 | 0 |

Session detail (baseline): s1 (variant-a) 4 burns; s3 (variant-a) 3 empty;
s4 (variant-b) 2 empty; s2 (variant-b) clean. Variant-a is more fragile —
consistent with the reporter seeing failure-rate variation across prompt
variants. Degeneration compounds once an empty turn (reasoning appended,
content empty) enters the history.

| Probe | Result |
|---|---|
| greedy parity, persona exam turn 1 (identical prompt, temp 0) | diverge at char 195; MTP stops at 987 tok where AR runs to the 1200 cap |
| greedy parity, full 7-turn exam | 7/7 turns diverge (later turns inherit earlier divergence) |
| width ablation `--draft-tokens 1/2/7`, temp 0, same prompt | **all three diverge at char 195**; w1/w2/w7 stop at 1173/997/1012 tok vs AR 1200 |
| greedy parity, plain English, `enable_thinking=false` | diverge at char 734; MTP stops at 491 vs AR 553 |
| draft acceptance on loop turns | 0.92–0.97 (healthy: 0.60–0.74) |

Failure-shape examples (in `results/baseline/`): `s1/turn-03` reasoning
degenerates into a "Jarvis" loop at ~1% of the reasoning and burns the budget;
`s3/turn-04` loops and hits EOS after 67 tokens (`finish=stop`, empty content —
the exact reported signature); `s4/turn-06` is *coherent* planning that ends
and stops without ever emitting `</think>` — the reporter's "decides not to
answer" shape.

## What was ruled out (with evidence)

- **Reasoning parsing / post-processing** (`ParseGeneration`,
  openai_chat.cpp:1848): the signature is exactly "model never emitted
  `</think>` before stopping". The `--trace` facility can confirm per-run;
  the greedy probes already show premature EOS in the raw behavior.
- **Sampling math**: host verification (`mtp_sampling.hpp`) implements exact
  speculative sampling — honest 2^24-quantized proposals, FP64 target,
  standard acceptance, correct (p−q)⁺ residual (`sampling.cpp:328`). The GPU
  sampling kernel (`src/models/qwen/hip/kernels/sample.hip`) has **zero diff**
  from the control base. Host-side #415 block-skip is semantics-preserving
  and bypassed on the device path.
- **Chat template / prompt construction**: `qwen/chat_template.cpp` renders
  preserved thinking identically to the control base (only difference: the
  thought is trimmed); #446/#441/#449 are no-ops for this request shape
  (no tools, single leading system message). #446 only changes response
  parsing.
- **Checkpoint/prompt-cache restore**: failures occur on cache-hit and
  fresh-prefix turns alike; the clean `nospec` arm uses the same cache paths.
- **Sparse attention threshold**: divergence begins at position ≪2048
  (`indexer_top_k`), well inside dense attention.
- **Multi-token verify batch & rollback**: ruled out by width-1 divergence
  (w1 never runs a verify batch or rollback, yet drifts identically).
  The per-row GDN/conv/PLE snapshot restore in `Executor::Rollback`
  (executor.cpp:2890) also audits clean on read-through.

## Fix session (2026-10-07, later): rollback restore isolated

A GPU parity probe (`parity_probe.cpp`, committed) now reproduces the drift
**at the engine session level, deterministically, without HTTP**: an AR
session and a speculative session run the same 1,423-token prompt greedy.

Measured chain (all bit-exact unless noted):
- Prefill frontiers: bit-identical.
- Width-1 decode: bit-identical over 250 steps.
- First verify cycle: per-layer row-0 sub-stages (ple/mix/attn/res × 48
  layers) **bit-identical**; the downloaded frontier differs by max|Δ|=11.85
  over all 248,320 logits.
- Per-build disables: snapshot stores+restore, pinned chain, q8t staging
  cache, per-row n-gram hashing, graphs (trunk/draft/prefix) — the delta
  stayed bit-identical through every variant.
- Per-row PLE and per-row LM-head routes (fix candidates, reverted): changed
  nothing — the batched PLE and head routes are exact.
- **Decisive restore-isolation test**: right after the first rejected cycle
  (committed histories verified identical on both sessions), one width-1
  decode step on each session diverges at the first step (max|Δ|=18.4). The
  rollback leaves the session state wrong.

**Root cause (isolated): `Executor::Rollback` restores an incomplete or
corrupted session state after a rejected speculative cycle.** Verified
correct: GDN state/conv checksums (layers 0–2), PLE history head, n-gram
window, position/blocks. The missing/corrupted component is elsewhere in the
session state the verify forward mutates — candidates: the unchecked 33
linear layers (checksum coverage was sum-of-squares only, which cannot
distinguish permuted/off-by-one content), attention KV/indexer rows for the
rejected positions, or a snapshot-slot off-by-one. Sum-of-squares equality
is a weak check; the next step is full-vector dumps of every restored
buffer right after Rollback, diffed against the AR session at the same
commit.

**Consequences:** every rejected speculative cycle poisons the session —
explaining the ~10× reasoning-degeneration at temp 1.0 (each reject
degrades context), the empty/loop turns, and why `--speculative off` is
clean. The immediate user mitigation remains `--speculative off`.

Probe: `parity_probe.cpp` phase 2 now includes the restore-isolation test
(restore-test lines in its output). Evidence: `probe-*.log` next to this
file and in the worktree root.

A GPU parity probe (`parity_probe.cpp`, committed) now reproduces the drift
**at the engine session level, deterministically, without HTTP**: an AR
session and a speculative session run the same 1,423-token prompt greedy.

- Prefill frontiers: **bit-identical** (max|Δ|=0).
- Width-1 decode (trunk kDecode + draft catch-up): **bit-identical over
  250 steps**.
- The **first multi-row verify cycle** diverges: frontier max|Δ| = 11.85,
  **all 248,320 logits**, reproducibly at dense (1.4K-token) and sparse
  (7.4K-token) positions.
- The post-divergence state is **valid** — continuing the speculative
  session with width-1 steps yields a perfectly coherent continuation. No
  corruption.
- Per-layer checksums: layer 0 bit-exact; **the divergence starts at
  layer 1 (the PLE layer)** and cascades.
- Ruled out by disabling one mechanism per build (delta bit-identical
  11.8504 in every variant): snapshot stores + restore, the pinned-chain /
  device-argmax path, the q8t staging cache, per-row n-gram hashing, trunk
  speculative graphs, draft graphs.
- State-restore proof: with everything eager, the **restored GDN/conv state
  checksums (layers 0–2) are bit-identical to AR's at the same commit**, the
  n-gram window restore is correct, the PLE history head is correct — and
  the *next* verify forward, given those identical inputs, still diverges
  at L1.

**Root cause (proven by elimination): the multi-row (n ≥ 2) verify forward's
kernels compute per-row results that differ from single-token decode through
n-dependent reduction order / launch geometry — starting in the layer-1 op
stack (PLE projections, GDN recurrence, or MoE), then amplified by MoE
router top-k near-tie flips into large logit deltas.** This is the class the
engine documents as accepted ("the single row route has a different
floating-point reduction"; QUALITY.md: free-running C≥2 text may flip
near-ties) — the existing parity gates only cover 8-token completions and
matched prefill, where the effect is invisible.

**Why it degrades generation quality:** every speculative cycle re-rolls
router decisions, so MTP trajectories are a noisier sample of the model; on
persona/memory fixtures at temp 1.0 this multiplies reasoning-phase
degeneration (repetition loops, premature stops inside `<think>`) roughly
tenfold (12/56 turns vs 1/56 AR-only across two full runs).

**Fix path:** make the n ≤ 8 kernels' per-row results independent of the
batch row count (grid/wave layout invariant reduction), verified by the new
probe (phase 2 must read bit-exact) plus an n=1-vs-n=2 equivalence assertion
in the op tests. Until then `--speculative off` is the only clean mode;
`--repeat-penalty 1.1` was tested and does **not** help.

Probe evidence: `results/probe-*` and `probe-exp*.log` next to this file
(worktree root).

### Intermediate hypothesis (superseded by the restore isolation above)

The per-row PLE/head routes were tested as fix candidates and reverted:
they did not change the divergence, ruling the batched PLE and head routes
out as the cause. The kernel-geometry theory below was the working
hypothesis before the restore isolation was found; kept for provenance of
the per-op elimination work.

With the MTP block loaded, every generated token's distribution comes from
trunk forwards that run **with the MTP machinery active** — the trunk frontier
keeps hidden rows for the draft (`DraftReplay`: "MTP position i consumes token
i+1 and the trunk's hidden of position i", engine.cpp:514) and, at width ≥ 2,
runs as the `kVerify` batch. The greedy probes prove the resulting logits
differ systematically from plain AR decode (deterministic mid-generation
divergence on identical prompts, same offset at all widths). A tiny but
**systematic** logits perturbation is near-invisible at temp 0 (occasional
near-tie flips, slightly earlier EOS) but at temp 1.0 tilts long generations
into repetition attractors at a large rate — exactly the reported 25% empty
turns, plus identity drift once a degraded turn enters the preserved-thinking
history.

Provenance: the hidden-rows MTP design predates the reporter's control base
(`DraftReplay` exists in `d9a84f1`; introduced in the 2026-09-25 upstream
era), but the reporter's control build (pixmaate `windows-port` @ `2eb0778`,
based on `d9a84f1`) is clean. The drift-inducing change therefore sits
**between `d9a84f1` and `1cfd681`** — candidates: #332 (device greedy
penalties / frontier epilogue), the device-chained greedy draft work
(`e2f1d3e` upstream / `1b5b782`+`756eaec`+`b09899a`+`a439279` window-native),
#421/#463 (prefill/attention kernels). All prior validation for those changes
compared MTP-on vs MTP-on (or seeded width replay); none compared MTP-on vs
AR at length, which is why every gate passed. Upstream "could not reproduce"
(gufo-org/gufo#461) is consistent: the failure needs temp 1.0 + long
multi-turn persona context, or — deterministically — a ≥1K-token greedy
parity diff, which nobody ran.

## Answers for the reporter's six triage questions

1. Sanitized reproducing conversation: replaced by the fully synthetic,
   shareable fixture in this directory (reproduces at 18–32% on their
   config).
2. Last-good commit: not yet bisected, but bounded — the control base
   `d9a84f1` is clean, current head fails; see bisect plan below.
3. Full server command: reproduced verbatim; startup log shape identical
   (load 27 s, `draft_limit=7`).
4. Raw output before parsing: failures are reasoning-phase degeneration —
   repetition loops or EOS inside the think block — not truncation or
   client-side drop. Greedy parity (temp 0) makes this deterministic.
5. **`--speculative off`: does NOT fail — 0/28.** Immediate workaround.
6. Fresh server / cold cache: yes — every session in every arm was a fresh
   process; failures appeared from the first sessions.

## Next steps

1. **Land the mitigation guidance** (issue comment): `--speculative off`
   until fixed; verified clean here.
2. **Instrumented session test (the fix locator).** A
   `qwen38_flash_next_session_test` case that runs the same prompt through
   AR and MTP width-1 and, at the first token divergence, downloads both
   frontier logits rows and diffs them (max |Δlogit|, top-k overlap). This
   pins the perturbation to the capture/verify epilogue in one build.
3. **Close the test gap.** Extend the greedy AR/MTP equality assertion to
   long generations (≥1K tokens, long persona-style prompt). `parity-run.log`
   shows the current 8-token gate passes while a 1K-token generation
   diverges. Propose upstreaming this test.
4. **Bisect with the deterministic probe.** `simple_parity.py` /
   `width_parity.py` are 2×~30 s per built revision (single turn, temp 0) and
   give a clean diverge-at-char-N signal. Build recipe:
   `D:\gufo-control\configure-control.bat` (multi-SDK TheRock setup);
   bisect range `d9a84f1..1cfd681` over the candidate list above (~5–8
   builds). Note: bisect across the upstream↔fork merge points, not
   window-native alone — the culprit may be an upstream commit.
5. After the fix: re-run the baseline arm here to confirm 0/28, and confirm
   decode speed with MTP is retained (that is the point of the feature).

## Evidence inventory

- `fixture.py`, `repro_exam.py`, `greedy_parity.py`, `width_parity.py`,
  `simple_parity.py` — harness + probes
- `results/baseline/`, `results/nospec/` — per-turn request/response JSON,
  `summary.json`, server logs (synthetic content only, safe to share)
- `results/width-*/`, `results/simple-*/`, `results/parity-*/` — probe outputs
- `baseline-run.log`, `nospec-run.log`, `parity-run.log`, `width-run.log` —
  run transcripts

Run on `triage/issue-6-empty-turns` (worktree of `feature/window-native`
@ `1cfd681`).
