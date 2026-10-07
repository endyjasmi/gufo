"""Issue #6 triage: replay the 7-turn persona/memory exam against a gufo server.

Runs each session against a FRESH server process (matching the reporter's
cold-cache setup), classifies turns, and writes full JSON evidence.

Usage:
  python repro_exam.py --sessions 4 --label baseline --port 18201
  python repro_exam.py --sessions 4 --label nospec --port 18231 --no-speculative
  python repro_exam.py --sessions 4 --label nopreserve --port 18241 --no-preserve
  python repro_exam.py --sessions 2 --label trace --port 18251 --trace
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fixture import VARIANTS  # noqa: E402

import os
BUILD_DIR = Path(os.environ.get("GUFO_BUILD_DIR",
                            r"D:\gufouild\windows-release"))
GUFO = BUILD_DIR / "gufo.exe"
HF = Path.home() / ".cache" / "huggingface" / "hub"
SNAP = HF / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / (
    "38bb39ee97821de2c9009abb7e93950eec396e66")
MODEL = SNAP / "UD-Q4_K_XL" / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
MTP = SNAP / "MTP" / "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"
MMPROJ = SNAP / "mmproj-BF16.gguf"

DENIAL_PATTERNS = [
    r"(我不是|并非)Jarvis",
    r"(我不是|并非)阿福",
    r"我?只是(?:一个)?(?:AI|语言模型|程序)",
    r"我是(?:Qwen|通义|千问)",
    r"(?:没有|不曾拥有)(?:身份|真实身份)",
    r"(?:不要|别)把.*(?:记忆|上下文|注入).*(?:当成|当作|视为)",
    r"(?:虚构|编造|捏造)",
]

OUT_ROOT = Path(__file__).parent / "results"


def server_args(port: int, label: str, no_speculative: bool, trace: bool,
                repeat_penalty: float = 1.0) -> list[str]:
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
        "--temperature", "1.0", "--top-p", "0.95", "--top-k", "20", "--min-p", "0",
        "--mmproj", str(MMPROJ),
    ]
    if repeat_penalty != 1.0:
        args += ["--repeat-penalty", str(repeat_penalty),
                 "--repeat-last-n", "128"]
    if no_speculative:
        args += ["--speculative", "off"]
    else:
        args += ["--speculative", "mtp", "--mtp-model", str(MTP)]
    if trace:
        trace_path = OUT_ROOT / f"trace-{label}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        args += ["--trace", str(trace_path)]
    return args


def wait_ready(port: int, deadline_s: float = 240.0) -> bool:
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            time.sleep(2.0)
    return False


def chat(port: int, payload: dict, timeout: float = 600.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def classify(content: str, reasoning: str, finish: str) -> list[str]:
    flags = []
    if finish == "stop" and content.strip() == "" and reasoning.strip():
        flags.append("EMPTY")
    for pattern in DENIAL_PATTERNS:
        if re.search(pattern, content):
            flags.append(f"DENIAL({pattern})")
            break
    return flags


def run_session(port: int, exam, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "system", "content": exam.system}]
    summary = {"variant": exam.name, "turns": []}
    for index, user_text in enumerate(exam.turns, start=1):
        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": "flash-next",
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "stream": False,
            "chat_template_kwargs": {"preserve_thinking": True},
        }
        started = time.time()
        try:
            response = chat(port, payload)
        except Exception as exc:  # noqa: BLE001 - record and continue
            summary["turns"].append({"turn": index, "error": repr(exc),
                                     "elapsed_s": round(time.time() - started, 1)})
            print(f"  turn {index}: ERROR {exc!r}", flush=True)
            continue
        elapsed = round(time.time() - started, 1)
        choice = response["choices"][0]
        message = choice["message"]
        finish = choice.get("finish_reason")
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        usage = response.get("usage", {})
        flags = classify(content, reasoning, finish)
        turn_record = {
            "turn": index, "finish": finish, "content_chars": len(content),
            "reasoning_chars": len(reasoning),
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "elapsed_s": elapsed, "flags": flags,
            "content_preview": content[:180],
            "reasoning_preview": reasoning[:180],
        }
        summary["turns"].append(turn_record)
        (out_dir / f"turn-{index:02d}.json").write_text(
            json.dumps({"request_messages": messages[:-1],
                        "request_user": user_text[-500:],
                        "response": response}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        # Append the assistant reply to history: empty turns append reasoning
        # instead, mirroring the reporter's client behavior.
        assistant = {"role": "assistant", "content": content}
        if content.strip() == "" and reasoning:
            assistant["reasoning_content"] = reasoning
        messages.append(assistant)
        marker = " ".join(flags) if flags else "ok"
        print(f"  turn {index}: finish={finish} content={len(content)}ch "
              f"reasoning={len(reasoning)}ch out={usage.get('completion_tokens')}tok "
              f"{elapsed}s [{marker}]", flush=True)
    return summary


def kill_server(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                   capture_output=True, check=False)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--port", type=int, default=18201)
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument("--no-preserve", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    cli = parser.parse_args()

    out_dir = OUT_ROOT / cli.label
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "server.log"

    variant_keys = ["a", "b", "a", "b", "a", "b", "a", "b"]
    overall = []
    for session in range(1, cli.sessions + 1):
        variant_key = variant_keys[(session - 1) % len(variant_keys)]
        exam = VARIANTS[variant_key]()
        print(f"== session {session} ({exam.name}) ==", flush=True)
        server_log = open(log_path.with_suffix(
            f".s{session}.log"), "w", encoding="utf-8")
        args = server_args(cli.port, cli.label, cli.no_speculative, cli.trace,
                           cli.repeat_penalty)
        if cli.no_preserve:
            index = args.index("--preserve-thinking")
            args[index + 1] = "off"
        proc = subprocess.Popen(args, cwd=BUILD_DIR, stdout=server_log,
                                stderr=subprocess.STDOUT)
        try:
            if not wait_ready(cli.port):
                print(f"server failed to become ready; see "
                      f"{log_path.with_suffix(f'.s{session}.log')}", flush=True)
                kill_server(proc)
                return 1
            summary = run_session(cli.port, exam, out_dir / f"s{session}")
        finally:
            kill_server(proc)
            server_log.close()
        empties = sum(1 for t in summary["turns"] if "EMPTY" in t.get("flags", []))
        denials = sum(1 for t in summary["turns"]
                      if any(f.startswith("DENIAL") for f in t.get("flags", [])))
        summary["empty_turns"] = empties
        summary["denial_turns"] = denials
        overall.append(summary)
        print(f"session {session}: empty={empties}/7 denial={denials}", flush=True)

    total_turns = sum(len(s["turns"]) for s in overall)
    total_empty = sum(s["empty_turns"] for s in overall)
    total_denial = sum(s["denial_turns"] for s in overall)
    print(f"\n== {cli.label}: empty {total_empty}/{total_turns} "
          f"({100.0 * total_empty / max(total_turns, 1):.1f}%), "
          f"denial {total_denial} ==", flush=True)
    (out_dir / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
