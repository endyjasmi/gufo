#!/usr/bin/env python3
"""Compare two or more `gufo bench --logit-eval` dumps bit for bit.

Dumps share the GFLE v1 format: a 16-byte header (magic, version, vocab,
top-k) followed by one record per teacher-forced position (position, width,
target token, top-1 token, target log-prob, log-sum-exp, then the top-k
token ids and their log-probabilities). Any build that writes this format
compares against any other, including reference forks.

Usage
-----
  tools/bench/logit_eval.py dumpA.bin dumpB.bin [dumpC.bin ...]

Each pair (first dump vs every other) reports:

  * positions compared and schedule width mix;
  * perplexity of each dump and its delta;
  * max |delta| of the target log-prob and of the log-sum-exp;
  * top-1 and top-64 agreement;
  * the first position where the dumps differ and by how much.

A dump pair is either "BIT-IDENTICAL" (every shared float equal) or reports
the smallest divergence; anything above exact equality but below ~1e-6 is
rounding-level, anything above ~0.1 changes sampled behavior at temperature 1.
"""

import math
import struct
import sys

MAGIC = b"GFLE"
HEADER = struct.Struct("<4sIII")
FIXED = struct.Struct("<iiiiff")


def read_dump(path):
    with open(path, "rb") as handle:
        data = handle.read()
    magic, version, vocab, top = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        sys.exit(f"{path}: not a GFLE dump (magic {magic!r})")
    if version != 1:
        sys.exit(f"{path}: unsupported GFLE version {version}")
    offset = HEADER.size
    entries = []
    stride = FIXED.size + 2 * top * 4
    while offset < len(data):
        if offset + stride > len(data):
            sys.exit(f"{path}: truncated record at byte {offset}")
        position, width, target, argmax, target_lp, lse = FIXED.unpack_from(
            data, offset
        )
        ids_off = offset + FIXED.size
        ids = struct.unpack_from(f"<{top}i", data, ids_off)
        lps = struct.unpack_from(f"<{top}f", data, ids_off + top * 4)
        entries.append(
            {
                "position": position,
                "width": width,
                "target": target,
                "argmax": argmax,
                "target_lp": target_lp,
                "lse": lse,
                "ids": ids,
                "lps": lps,
            }
        )
        offset += stride
    return {"vocab": vocab, "top": top, "entries": entries}


def perplexity(entries):
    nll = -sum(entry["target_lp"] for entry in entries)
    return math.exp(nll / len(entries))


def compare(base_path, base, other_path, other):
    if base["vocab"] != other["vocab"] or base["top"] != other["top"]:
        sys.exit(
            f"{other_path}: vocab/top mismatch "
            f"({other['vocab']}/{other['top']} vs "
            f"{base['vocab']}/{base['top']})"
        )
    by_position = {entry["position"]: entry for entry in other["entries"]}
    compared = 0
    target_delta = 0.0
    lse_delta = 0.0
    top1_agree = 0
    top64_agree = 0
    bit_identical = True
    first_diff = None
    for entry in base["entries"]:
        twin = by_position.get(entry["position"])
        if twin is None:
            continue
        compared += 1
        if twin["target_lp"] != entry["target_lp"] or twin["lse"] != entry["lse"]:
            bit_identical = False
        dt = abs(twin["target_lp"] - entry["target_lp"])
        dl = abs(twin["lse"] - entry["lse"])
        if dt > target_delta:
            target_delta = dt
        if dl > lse_delta:
            lse_delta = dl
        if twin["argmax"] == entry["argmax"]:
            top1_agree += 1
        elif first_diff is None:
            first_diff = (
                entry["position"],
                entry["argmax"],
                twin["argmax"],
                entry["target_lp"],
                twin["target_lp"],
            )
        if twin["ids"] == entry["ids"]:
            top64_agree += 1
    if compared == 0:
        sys.exit(f"{other_path}: no overlapping positions with {base_path}")
    print(f"\n{base_path}  vs  {other_path}")
    print(f"  positions compared : {compared}")
    print(
        f"  perplexity         : {perplexity(base['entries']):.6f} vs "
        f"{perplexity(other['entries']):.6f} "
        f"(delta {perplexity(other['entries']) - perplexity(base['entries']):+.2e})"
    )
    print(f"  max |d target lp|  : {target_delta:.3e}")
    print(f"  max |d lse|        : {lse_delta:.3e}")
    print(f"  top-1 agreement    : {top1_agree}/{compared}")
    print(f"  top-64 agreement   : {top64_agree}/{compared}")
    if bit_identical:
        print("  verdict            : BIT-IDENTICAL")
    else:
        print("  verdict            : DIVERGES")
        if first_diff is not None:
            position, a, b, lpa, lpb = first_diff
            print(
                f"  first top-1 flip   : position {position}, "
                f"token {a} (lp {lpa:.6f}) vs token {b} (lp {lpb:.6f})"
            )


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    base_path = argv[1]
    base = read_dump(base_path)
    print(
        f"{base_path}: {len(base['entries'])} positions, vocab "
        f"{base['vocab']}, top-{base['top']}"
    )
    for other_path in argv[2:]:
        compare(base_path, base, other_path, read_dump(other_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
