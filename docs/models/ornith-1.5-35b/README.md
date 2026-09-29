# Ornith-1.5-35B (qwen35moe)

Ornith-1.5-35B-A3B is a 35B-parameter (3B active) hybrid text model in the
Qwen3.5-MoE lineage: 40 layers alternating Gated DeltaNet linear attention and
full gated GQA attention (every fourth layer), 256 routed experts with 8 active
plus one shared expert, and a multi-token-prediction draft block stored inside
the model file (`nextn`). Gufo registers the architecture as `qwen35moe`.

## Acquisition

```sh
hf download ornith-ai/Ornith-1.5-35B-A3B-GGUF   Ornith-1.5-35B-Q4_K_M.gguf --local-dir models/ornith-1.5-35b
```

MIT licensed; the GGUF embeds the tokenizer and chat template. Optional vision
sidecar: `mmproj-Ornith-1.5-35B-BF16.gguf` (auto-discovered beside the model).

## Usage

```sh
gufo prompt -m models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf -p "Hello"
gufo serve llm -m models/ornith-1.5-35b/Ornith-1.5-35B-Q4_K_M.gguf
gufo bench -m ... -p 2048 -n 128 -r 2
```

Speculative decoding (`--speculative mtp`) uses the in-file draft block; no
sidecar or `--mtp-model` is needed.

## Notes

- Dense Q4_K/Q6_K projections are host-dequantized to Q8_0 at load; routed
  Q4_K experts keep their native layout on the MMQ/WMMA tiers.
- The current Windows port qualifies this model on a single Strix Halo
  gfx1151 host; Linux numbers land in BENCHMARKS.md once measured.
