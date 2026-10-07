"""Issue #6 fix A/B benchmark: prompt processing and decode, AR and MTP, C1/C2.

Compares two builds of the same commit (baseline = pristine parent, fixed =
per-row PLE/head fix) on the reporter's model, greedy, thinking off — the
BENCHMARKS.md method as closely as a standalone driver can.

Matrix per build: {ar, mtp} x {c1, c2} x {pp2048, tg128}.
Per request the server reports prompt_tokens_per_second /
completion_tokens_per_second / prefill_ms / decode_ms (usage.gufo); this
driver records every sample, never averages blindly.

Usage: python bench_bilateral.py --out results/bench-fix.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HF = Path.home() / ".cache" / "huggingface" / "hub"
SNAP = HF / "models--unsloth--Qwen3.8-Flash-Next-GGUF" / "snapshots" / (
    "38bb39ee97821de2c9009abb7e93950eec396e66")
MODEL = SNAP / "UD-Q4_K_XL" / "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
MTP = SNAP / "MTP" / "mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"

BUILDS = {
    "baseline": Path(r"D:\gufo\build\windows-release"),
    "fixed": Path(r"D:\gufo-issue6\build\windows-release"),
}

FILLER = (
    "项目背景：胖哥的私人助理系统迁移到新的集群环境，涉及 GPU 调度、接口文档、"
    "报销流程和小雨的科学展准备。Each milestone is tracked on the board; the "
    "tracking board lists owners, dates and blockers for the migration. "
    "记忆卡系统按日期归档，卡片内容包括约定事项、家庭事务与工作协同的细节。 "
    "The staging environment mirrors production configuration, so the demo "
    "checks run against the same routing rules and the same quota limits. "
)


def prompt_tokens(target: int) -> str:
    # ~1.35 tokens per CJK char / ~0.75 per EN word in this mix; scale then
    # let the server report the real count.
    return (FILLER * ((target // 55) + 1))[: target * 3]


def server_args(build: Path, mode: str, conc: int, port: int) -> list[str]:
    args = [
        str(build / "gufo.exe"), "serve",
        "--host", "127.0.0.1", "--port", str(port),
        "--sessions", str(conc), "--max-request-bytes", "33554432",
        "llm",
        "--model", str(MODEL),
        "--served-model-name", "flash-next",
        "--context", "32768",
        "--max-output-bytes", "8388608",
        "--think", "off",
        "--temperature", "0",
    ]
    if mode == "mtp":
        args += ["--speculative", "mtp", "--mtp-model", str(MTP)]
    else:
        args += ["--speculative", "off"]
    return args


def wait_ready(port: int, deadline_s: float = 240.0) -> bool:
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except OSError:
            time.sleep(2.0)
    return False


def chat(port: int, user_text: str, max_tokens: int) -> dict:
    payload = {
        "model": "flash-next",
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_point(port: int, conc: int, kind: str, tag: str) -> dict:
    """One measured point: warmup + samples, sequential or concurrent."""
    samples = []
    locks = threading.Barrier(conc) if conc > 1 else None

    def one(index: int, out: list) -> None:
        text = prompt_tokens(2048) + f"\nVariation {tag}-{index}."
        max_tokens = 16 if kind == "pp2048" else 128
        if kind == "tg128":
            text = f"Variation {tag}-{index}. 请继续描述迁移计划。"
        if locks:
            locks.wait()
        response = chat(port, text, max_tokens)
        g = response["usage"].get("gufo", {})
        u = response["usage"]
        out.append({
            "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"),
            "pp_tok_s": u.get("prompt_tokens_per_second"),
            "tg_tok_s": u.get("completion_tokens_per_second"),
            "prefill_ms": g.get("prefill_ms"),
            "decode_ms": g.get("decode_ms"),
            "draft_accepted": u.get("draft_tokens_accepted"),
            "draft_tokens": u.get("draft_tokens"),
            "cache_hit": g.get("cache_hit"),
        })

    # warmup
    warm = []
    threads = [threading.Thread(target=one, args=(90 + i, warm))
               for i in range(conc)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for sample in range(3):
        batch = []
        threads = [threading.Thread(target=one,
                                    args=(sample * conc + i, batch))
                   for i in range(conc)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        samples.extend(batch)
        print(f"    {kind}/{tag} sample {sample}: " + " | ".join(
            f"pp={s['pp_tok_s'] and round(s['pp_tok_s'], 1)} "
            f"tg={s['tg_tok_s'] and round(s['tg_tok_s'], 2)} "
            f"(ptok={s['prompt_tokens']}, out={s['completion_tokens']}, "
            f"cache={s['cache_hit']})" for s in batch), flush=True)
    return {"kind": kind, "tag": tag, "conc": conc, "samples": samples}


def kill(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                   capture_output=True, check=False)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/bench-fix.json")
    parser.add_argument("--port-base", type=int, default=18310)
    cli = parser.parse_args()

    results = []
    port = cli.port_base
    for build_name, build in BUILDS.items():
        for mode in ("ar", "mtp"):
            for conc in (1, 2):
                port += 1
                label = f"{build_name}/{mode}/c{conc}"
                print(f"== {label} (port {port}) ==", flush=True)
                log = open(Path(__file__).parent / f"bench-{build_name}-"
                           f"{mode}-c{conc}.log", "w", encoding="utf-8")
                proc = subprocess.Popen(server_args(build, mode, conc, port),
                                        cwd=build, stdout=log,
                                        stderr=subprocess.STDOUT)
                try:
                    if not wait_ready(port):
                        print(f"  server failed: {label}", flush=True)
                        continue
                    time.sleep(2)
                    for kind in ("pp2048", "tg128"):
                        point = run_point(port, conc, kind, label)
                        point.update(build=build_name, mode=mode)
                        results.append(point)
                finally:
                    kill(proc)
                    log.close()
                    time.sleep(3)

    out_path = Path(__file__).parent / cli.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")

    # summary: median per point
    print("\n==== summary (median per request; C2 aggregate = sum of the "
          "2 concurrent medians) ====")
    import statistics
    for mode in ("ar", "mtp"):
        for conc in (1, 2):
            for kind in ("pp2048", "tg128"):
                row = [f"{mode.upper()} C{conc} {kind}:"]
                for build_name in BUILDS:
                    pts = [s for p in results
                           if p["build"] == build_name and p["mode"] == mode
                           and p["conc"] == conc and p["kind"] == kind
                           for s in p["samples"] if s["cache_hit"] is False]
                    key = "pp_tok_s" if kind == "pp2048" else "tg_tok_s"
                    vals = [s[key] for s in pts if s[key]]
                    if vals:
                        row.append(f"{build_name}={statistics.median(vals):.2f}"
                                   f" (n={len(vals)})")
                    else:
                        row.append(f"{build_name}=n/a")
                print("  " + "  ".join(row), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
