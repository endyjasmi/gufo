"""Four-way greedy width ablation on the IDENTICAL turn-1 prompt.

  ar      : --speculative off                    (reference)
  mtp-w1  : MTP loaded, --draft-tokens 1         (machinery active, no multi-token verify)
  mtp-w2  : --draft-tokens 2                     (minimal speculation)
  mtp-w7  : default --draft-tokens 7             (full speculation)

All four use temp 0 / greedy, same fixture turn 1, same max_tokens.
Comparing outputs localizes the divergence source.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fixture import VARIANTS  # noqa: E402
from repro_exam import (BUILD_DIR, GUFO, MODEL, MMPROJ, MTP, kill_server,
                        wait_ready)  # noqa: E402

OUT_ROOT = Path(__file__).parent / "results"
PORT = 18271


def server_args(width: int | None) -> list[str]:
    args = [
        str(GUFO), "serve",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--sessions", "1", "--max-request-bytes", "33554432",
        "llm",
        "--model", str(MODEL),
        "--served-model-name", "flash-next",
        "--context", "262144",
        "--max-output-bytes", "8388608",
        "--think", "on",
        "--reasoning-effort", "medium",
        "--preserve-thinking", "on",
        "--temperature", "0", "--top-p", "1", "--top-k", "0", "--min-p", "0",
        "--mmproj", str(MMPROJ),
    ]
    if width is None:
        args += ["--speculative", "off"]
    else:
        args += ["--speculative", "mtp", "--mtp-model", str(MTP),
                 "--draft-tokens", str(width)]
    return args


def main() -> int:
    exam = VARIANTS["a"]()
    messages = [{"role": "system", "content": exam.system},
                {"role": "user", "content": exam.turns[0]}]
    payload = {
        "model": "flash-next",
        "messages": messages,
        "max_tokens": 1200,
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "stream": False,
        "chat_template_kwargs": {"preserve_thinking": True},
    }
    results = {}
    for label, width in (("ar", None), ("mtp-w1", 1), ("mtp-w2", 2),
                         ("mtp-w7", 7)):
        out_dir = OUT_ROOT / f"width-{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "server.log", "w", encoding="utf-8")
        proc = subprocess.Popen(server_args(width), cwd=BUILD_DIR,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            if not wait_ready(PORT):
                raise RuntimeError(f"server ({label}) failed to start")
            started = time.time()
            response = json.loads(urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}),
                timeout=900.0).read())
            choice = response["choices"][0]
            results[label] = {
                "finish": choice.get("finish_reason"),
                "reasoning": choice["message"].get("reasoning_content") or "",
                "content": choice["message"].get("content") or "",
                "completion_tokens": response["usage"]["completion_tokens"],
                "draft": (response["usage"].get("draft_tokens_accepted", 0),
                          response["usage"].get("draft_tokens", 0)),
                "elapsed_s": round(time.time() - started, 1),
            }
            r = results[label]
            print(f"[{label}] {r['finish']} {r['completion_tokens']}tok "
                  f"acc={r['draft'][0]}/{r['draft'][1]} {r['elapsed_s']}s",
                  flush=True)
        finally:
            kill_server(proc)
            log.close()
        (out_dir / "response.json").write_text(
            json.dumps(results[label], ensure_ascii=False, indent=1),
            encoding="utf-8")

    reference = results["ar"]
    combined_ref = reference["reasoning"] + "\x00" + reference["content"]
    print(f"\nreference (ar): {reference['finish']} "
          f"{reference['completion_tokens']}tok")
    for label in ("mtp-w1", "mtp-w2", "mtp-w7"):
        r = results[label]
        combined = r["reasoning"] + "\x00" + r["content"]
        if combined == combined_ref:
            print(f"{label}: IDENTICAL to ar")
            continue
        cut = next((i for i, (x, y) in enumerate(zip(combined, combined_ref))
                    if x != y), min(len(combined), len(combined_ref)))
        print(f"{label}: DIVERGES at char {cut} "
              f"(mtp {r['completion_tokens']}tok vs ar "
              f"{reference['completion_tokens']}tok)")
        print(f"   mtp: ...{combined[max(0, cut - 50):cut + 50]!r}")
        print(f"   ar : ...{combined_ref[max(0, cut - 50):cut + 50]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
