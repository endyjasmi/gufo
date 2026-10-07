"""Greedy parity on a plain prompt (thinking off): universal drift or
persona-exam-specific? One server per arm, single identical prompt."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from repro_exam import (BUILD_DIR, GUFO, MODEL, MMPROJ, MTP, kill_server,
                        wait_ready)  # noqa: E402

OUT_ROOT = Path(__file__).parent / "results"
PORT = 18281

PROMPT = ("Write a detailed 350-word explanation of how a heat pump works, "
          "in plain English.")


def server_args(mtp: bool) -> list[str]:
    args = [
        str(GUFO), "serve",
        "--host", "127.0.0.1", "--port", str(PORT),
        "--sessions", "1",
        "llm",
        "--model", str(MODEL),
        "--served-model-name", "flash-next",
        "--context", "32768",
        "--max-output-bytes", "8388608",
        "--think", "on",
        "--reasoning-effort", "medium",
        "--temperature", "0", "--top-p", "1", "--top-k", "0", "--min-p", "0",
    ]
    if mtp:
        args += ["--speculative", "mtp", "--mtp-model", str(MTP)]
    else:
        args += ["--speculative", "off"]
    return args


def main() -> int:
    payload = {
        "model": "flash-next",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 700,
        "temperature": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    results = {}
    for label, mtp in (("ar", False), ("mtp", True)):
        out_dir = OUT_ROOT / f"simple-{label}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "server.log", "w", encoding="utf-8")
        proc = subprocess.Popen(server_args(mtp), cwd=BUILD_DIR, stdout=log,
                                stderr=subprocess.STDOUT)
        try:
            if not wait_ready(PORT):
                raise RuntimeError(f"server ({label}) failed to start")
            response = json.loads(urllib.request.urlopen(
                urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}),
                timeout=600.0).read())
            choice = response["choices"][0]
            results[label] = (choice["message"].get("content") or "",
                              response["usage"]["completion_tokens"])
            print(f"[{label}] {choice['finish_reason']} "
                  f"{results[label][1]}tok", flush=True)
        finally:
            kill_server(proc)
            log.close()
        (out_dir / "response.json").write_text(json.dumps(
            {"content": results[label][0]}, indent=1), encoding="utf-8")

    ar, mtp = results["ar"][0], results["mtp"][0]
    if ar == mtp:
        print("\nsimple-prompt greedy parity: IDENTICAL")
    else:
        cut = next((i for i, (x, y) in enumerate(zip(mtp, ar)) if x != y),
                   min(len(mtp), len(ar)))
        print(f"\nsimple-prompt greedy parity: DIVERGES at char {cut}")
        print(f"   mtp: ...{mtp[max(0, cut - 60):cut + 60]!r}")
        print(f"   ar : ...{ar[max(0, cut - 60):cut + 60]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
