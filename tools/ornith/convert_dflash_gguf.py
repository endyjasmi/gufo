#!/usr/bin/env python3
"""Convert a Hugging Face DFlash (v1) draft checkpoint to the Gufo dflash GGUF.

The Ornith-1.5-35B-A3B-DFlash checkpoint (ornith-ai/Ornith-1.5-35B-A3B-DFlash)
is a plain block-diffusion draft: a fused feature encoder (`fc` + `hidden_norm`),
a small Qwen3-style transformer over the masked block, and an output norm. It
has no dynamic convolutions or candidate selector (those are DFlash-2 additions)
and no embedding or LM head: the draft borrows the target model's.

Gufo's `dflash` GGUF keeps the same tensor names the DFlash-2 converter used,
marks the absent pieces through metadata, and encodes per-layer causality
(v1 sliding-attention layers are causal inside the block; the trailing
full-attention layers are not).
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
GGUF_VERSION = 3
ALIGNMENT = 32

TYPE_F32 = 0
TYPE_BF16 = 30
TYPE_Q8_0 = 8

META_STRING = 8
META_ARRAY = 9
META_UINT32 = 4
META_UINT64 = 10
META_FLOAT32 = 6
META_BOOL = 7


def write_string(out: bytearray, value: str) -> None:
    data = value.encode("utf-8")
    out += struct.pack("<Q", len(data))
    out += data


def write_metadata(out: bytearray, key: str, kind: int, value: bytes | int | float | str | list) -> None:
    write_string(out, key)
    out += struct.pack("<I", kind)
    if kind == META_STRING:
        write_string(out, value)
    elif kind == META_BOOL:
        out += struct.pack("<B", int(value))
    elif kind == META_UINT32:
        out += struct.pack("<I", int(value))
    elif kind == META_UINT64:
        out += struct.pack("<Q", int(value))
    elif kind == META_FLOAT32:
        out += struct.pack("<f", float(value))
    elif kind == META_ARRAY:
        elem_kind, elements = value
        out += struct.pack("<I", elem_kind)
        out += struct.pack("<Q", len(elements))
        for element in elements:
            if elem_kind == META_STRING:
                write_string(out, element)
            elif elem_kind == META_UINT32:
                out += struct.pack("<I", int(element))
            elif elem_kind == META_UINT64:
                out += struct.pack("<Q", int(element))
            else:
                raise ValueError(f"unsupported array element kind {elem_kind}")
    else:
        raise ValueError(f"unsupported metadata kind {kind}")


def align(offset: int) -> int:
    return (offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def quantize_q8_0(raw_bf16: np.ndarray) -> bytes:
    """llama.cpp Q8_0: per-32 blocks of f16 scale + int8 codes.

    `raw_bf16` holds raw BF16 bit patterns; decode to FP32 first.
    """
    weights = (raw_bf16.astype(np.uint32) << 16).view(np.float32)
    flat = weights.astype(np.float32).reshape(-1)
    blocks = (flat.size + 31) // 32
    flat = np.concatenate([flat, np.zeros(blocks * 32 - flat.size, dtype=np.float32)])
    blocks = flat.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    scale = amax / 127.0
    scale[amax == 0] = 1.0  # zero blocks keep zero codes with a unit scale
    codes = np.clip(np.rint(blocks / scale[:, None]), -127, 127).astype(np.int8)
    codes[amax == 0] = 0
    scale_f16 = scale.astype(np.float16).view(np.uint8).reshape(-1, 2)
    return b"".join(
        scale_f16[row].tobytes() + codes[row].tobytes() for row in range(blocks.shape[0])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="directory with config.json and model.safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--q8", action="store_true",
                        help="store dense draft matrices as Q8_0 (k/v stay BF16); "
                             "default keeps the checkpoint BF16")
    args = parser.parse_args()

    config = json.loads((args.checkpoint / "config.json").read_text())
    dflash = config["dflash_config"]

    weights_path = args.checkpoint / "model.safetensors"
    with weights_path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    data_offset = 8 + header_length
    tensors = {name: spec for name, spec in header.items() if name != "__metadata__"}

    layer_types = config["layer_types"]
    if len(layer_types) != config["num_hidden_layers"]:
        raise ValueError("layer_types does not match num_hidden_layers")
    if any(kind not in ("sliding_attention", "full_attention") for kind in layer_types):
        raise ValueError(f"unsupported draft layer types: {layer_types}")

    # The converter stores target taps as 1-based layer-input indices: the
    # config's target_layer_ids name layer outputs, and hidden_states[i] is the
    # input of layer i (so the output of layer N is hidden_states[N + 1]).
    target_layers = [layer + 1 for layer in dflash["target_layer_ids"]]
    layer_causal = [1 if kind == "sliding_attention" else 0 for kind in layer_types]

    metadata: list[tuple[str, int, object]] = [
        ("general.architecture", META_STRING, "dflash"),
        ("general.name", META_STRING, config.get("_name_or_path", "dflash-draft")),
        ("dflash.version", META_UINT32, 1),
        ("dflash.block_count", META_UINT32, config["num_hidden_layers"]),
        ("dflash.embedding_length", META_UINT32, config["hidden_size"]),
        ("dflash.feed_forward_length", META_UINT32, config["intermediate_size"]),
        ("dflash.attention.head_count", META_UINT32, config["num_attention_heads"]),
        ("dflash.attention.head_count_kv", META_UINT32, config["num_key_value_heads"]),
        ("dflash.attention.key_length", META_UINT32, config["head_dim"]),
        ("dflash.attention.layer_norm_rms_epsilon", META_FLOAT32, config["rms_norm_eps"]),
        ("dflash.rope.dimension_count", META_UINT32, config["head_dim"]),
        ("dflash.rope.freq_base", META_FLOAT32,
         config["rope_parameters"]["rope_theta"]),
        ("dflash.context_length", META_UINT32, config["max_position_embeddings"]),
        ("dflash.block_size", META_UINT32, dflash["block_size"]),
        ("dflash.attention.sliding_window", META_UINT32, config["sliding_window"]),
        ("dflash.attention.causal", META_BOOL, 0),
        ("dflash.target_layers", META_ARRAY, (META_UINT32, target_layers)),
        ("dflash.layer_causal", META_ARRAY, (META_UINT32, layer_causal)),
        ("dflash.vocab_size", META_UINT32, config["vocab_size"]),
        ("tokenizer.ggml.mask_token_id", META_UINT32, dflash["mask_token_id"]),
    ]

    # GGUF tensor names follow the DFlash-2 converter. HF stores [out, in]
    # row-major; GGUF ne[] lists the same bytes as [in, out].
    matrix_q8 = args.q8
    renames = {
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "enc.output_norm.weight",
        "norm.weight": "output_norm.weight",
    }
    per_layer = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
    }
    # These two are repacked to BF16 at load either way (the injection GEMM
    # path is BF16); keeping the file BF16 avoids a pointless requantize.
    always_bf16 = {"self_attn.k_proj.weight", "self_attn.v_proj.weight"}
    vectors = {
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_norm.weight", "self_attn.k_norm.weight",
    }

    plan: list[tuple[str, str, str]] = []  # (gguf name, source name, kind)
    for name in sorted(tensors):
        spec = tensors[name]
        if len(spec["shape"]) not in (1, 2):
            raise ValueError(f"unexpected tensor rank for {name}: {spec['shape']}")
        if name in renames:
            kind = "bf16"
            plan.append((renames[name], name, kind))
            continue
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "layers":
            raise ValueError(f"unmapped checkpoint tensor {name}")
        layer_index = parts[1]
        suffix = ".".join(parts[2:])
        if suffix not in per_layer:
            raise ValueError(f"unmapped checkpoint tensor {name}")
        is_matrix = len(spec["shape"]) == 2
        is_vector = suffix in vectors
        if is_matrix == is_vector:
            raise ValueError(f"tensor {name} does not match its expected rank")
        kind = "bf16"
        if is_matrix and matrix_q8 and suffix not in always_bf16:
            kind = "q8_0"
        plan.append((f"blk.{layer_index}.{per_layer[suffix]}", name, kind))

    # Tensor infos and raw payloads.
    infos: bytearray = bytearray()
    payload: bytearray = bytearray()
    offset = 0
    for gguf_name, source_name, kind in plan:
        spec = tensors[source_name]
        raw = _read_tensor(weights_path, spec, data_offset)
        if kind == "bf16":
            blob = raw.tobytes()
            ggml_type = TYPE_BF16
            ne = list(reversed(spec["shape"]))
        elif kind == "q8_0":
            blob = quantize_q8_0(raw)
            ggml_type = TYPE_Q8_0
            ne = list(reversed(spec["shape"]))
        else:
            raise AssertionError(kind)
        if offset % ALIGNMENT:
            payload += b"\0" * (ALIGNMENT - offset % ALIGNMENT)
            offset = align(offset)
        write_string(infos, gguf_name)
        infos += struct.pack("<I", len(ne))
        for dim in ne:
            infos += struct.pack("<Q", dim)
        infos += struct.pack("<I", ggml_type)
        infos += struct.pack("<Q", offset)
        payload += blob
        offset += len(blob)

    out = bytearray()
    out += struct.pack("<I", GGUF_MAGIC)
    out += struct.pack("<I", GGUF_VERSION)
    out += struct.pack("<Q", len(plan))
    out += struct.pack("<Q", len(metadata))
    for key, kind, value in metadata:
        write_metadata(out, key, kind, value)
    out += infos
    if len(out) % ALIGNMENT:
        out += b"\0" * (ALIGNMENT - len(out) % ALIGNMENT)
    out += payload

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(out)
    total_bytes = args.output.stat().st_size
    print(f"wrote {args.output} ({total_bytes / (1 << 20):.1f} MiB, "
          f"{len(plan)} tensors, q8={int(matrix_q8)})")


def _read_tensor(path: Path, spec: dict, data_offset: int) -> np.ndarray:
    dtype = spec["dtype"]
    if dtype != "BF16":
        raise ValueError(f"unsupported checkpoint dtype {dtype}; expected BF16")
    shape = spec["shape"]
    count = int(np.prod(shape))
    start = data_offset + spec["data_offsets"][0]
    end = data_offset + spec["data_offsets"][1]
    expected = count * 2
    if end - start != expected:
        raise ValueError(f"tensor {spec} has {end - start} bytes; expected {expected}")
    raw = np.fromfile(path, dtype="<u2", count=(end - start) // 2, offset=start)
    return raw.reshape(shape)


if __name__ == "__main__":
    sys.exit(main())
