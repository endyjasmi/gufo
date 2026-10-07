"""Greedy parity probe: MTP on vs off must produce identical tokens at temp 0.

Any divergence means the batched verification forward (kVerify) disagrees
with step-by-step decode (kDecode) on the same context -- a mode-dependent
numerics drift that sampled decoding then amplifies.

Runs one server per arm (temp 0 / greedy, seeded sampling config), replays
the same long multi-turn persona exam (single final generation), and diffs
the generated text.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fixture import VARIANTS  # noqa: E402
from repro_exam import (BUILD_DIR, GUFO, MODEL, MMPROJ, MTP, kill_server,
                        wait_ready)  # noqa: E402

OUT_ROOT = Path(__file__).parent / "results"


def server_args(port: int, mtp: bool) -> list[str]:
    args = [
        str(GUFO), "serve",
        "--host", "127.0.0.1", "--port", str(port),
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
    if mtp:
        args += ["--speculative", "mtp", "--mtp-model", str(MTP)]
    else:
        args += ["--speculative", "off"]
    return args


def chat(port: int, payload: dict, timeout: float = 900.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_arm(label: str, port: int, mtp: bool) -> dict:
    exam = VARIANTS["a"]()
    out_dir = OUT_ROOT / f"parity-{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": exam.system}]
    turns = []
    log = open(out_dir / "server.log", "w", encoding="utf-8")
    proc = subprocess.Popen(server_args(port, mtp), cwd=BUILD_DIR,
                            stdout=log, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(port):
            raise RuntimeError(f"server ({label}) failed to start")
        for index, user_text in enumerate(exam.turns, start=1):
            messages.append({"role": "user", "content": user_text})
            payload = {
                "model": "flash-next",
                "messages": messages,
                "max_tokens": 1200,
                "temperature": 0,
                "top_p": 1,
                "top_k": 0,
                "stream": False,
                "chat_template_kwargs": {"preserve_thinking": True},
                "seed": 42,
            }
            started = time.time()
            response = chat(port, payload)
            choice = response["choices"][0]
            message = choice["message"]
            usage = response.get("usage", {})
            turns.append({
                "turn": index,
                "finish": choice.get("finish_reason"),
                "content": message.get("content") or "",
                "reasoning": message.get("reasoning_content") or "",
                "completion_tokens": usage.get("completion_tokens"),
                "elapsed_s": round(time.time() - started, 1),
            })
            print(f"  [{label}] turn {index}: {choice.get('finish_reason')} "
                  f"{usage.get('completion_tokens')}tok "
                  f"{round(time.time() - started, 1)}s", flush=True)
            assistant = {"role": "assistant", "content": turns[-1]["content"]}
            if not turns[-1]["content"].strip() and turns[-1]["reasoning"]:
                assistant["reasoning_content"] = turns[-1]["reasoning"]
            messages.append(assistant)
    finally:
        kill_server(proc)
        log.close()
    (out_dir / "turns.json").write_text(
        json.dumps(turns, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"label": label, "turns": turns}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-base", type=int, default=18261)
    cli = parser.parse_args()
    mtp_result = run_arm("mtp", cli.port_base, True)
    ar_result = run_arm("ar", cli.port_base + 1, False)

    diffs = 0
    for m, a in zip(mtp_result["turns"], ar_result["turns"]):
        if m["content"] != a["content"] or m["reasoning"] != a["reasoning"]:
            diffs += 1
            print(f"turn {m['turn']}: DIVERGES (mtp {m['completion_tokens']}tok "
                  f"vs ar {a['completion_tokens']}tok)")
            for key in ("reasoning", "content"):
                mv, av = m[key], a[key]
                if mv != av:
                    cut = next((i for i, (x, y) in enumerate(zip(mv, av))
                                if x != y), min(len(mv), len(av)))
                    print(f"  {key}: first diff at {cut}\n"
                          f"    mtp: ...{mv[max(0, cut - 60):cut + 60]!r}\n"
                          f"    ar : ...{av[max(0, cut - 60):cut + 60]!r}")
        else:
            print(f"turn {m['turn']}: identical "
                  f"({m['completion_tokens']} vs {a['completion_tokens']} tok)")
    verdict = "IDENTICAL" if diffs == 0 else f"{diffs} TURN(S) DIVERGE"
    print(f"\ngreedy parity MTP-on vs MTP-off: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
