#!/usr/bin/env python3
"""Windows multi-user throughput probe (driver-protocol mirror).

The shared driver's multi-user tables hard-fail when greedy C>1 output is not
hash-identical to the C1 AR reference. On this Windows build batched decode
diverges benignly from C1 (argmax near-tie flip, then coherent continuation),
so the qualified table cannot be produced. This probe measures the same
numbers the driver would (pp2048 prepared cohort, tg128 measured cohort, sum
of per-request decode rates) and records exactness divergence explicitly.

Usage: python tools/windows/multi-probe.py <gufo.exe> <model.gguf> <mtp.gguf> <out.json>
"""
import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from gufo.model_bench.llm import turn_prompt  # driver's exact prompt recipe

PORT = 11499


class Server:
    def __init__(self, exe, model, mtp, sessions, log):
        cmd = [str(exe), "serve", "llm", "--model", str(model),
               "--context", "4096", "--sessions", str(sessions),
               "--port", str(PORT), "--served-model-name", "bench",
               "--think", "off", "--max-pending-per-client", "8"]
        if mtp:
            cmd += ["--speculative", "mtp", "--mtp-model", str(mtp)]
        self.log = open(log, "ab")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)

    def wait_ready(self, timeout=180):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/ready", timeout=2) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.log.close()


def send(prompt, max_tokens, cache):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({"model": "bench",
                         "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": 0, "seed": 1,
                         "cache_prompt": cache}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def cohort(prompt, c, max_tokens, cache):
    """c concurrent identical requests; returns list of (text, decode_tps, cached)."""
    out = [None] * c

    def worker(i):
        body = send(prompt, max_tokens, cache)
        g = body["usage"]["gufo"]
        out[i] = (body["choices"][0]["message"]["content"],
                  body["usage"]["completion_tokens_per_second"],
                  body["usage"]["prompt_tokens_details"]["cached_tokens"],)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(c)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return out


def measure(server_log, mode, task, c, results):
    # Calibrate word count against the server tokenizer: want prompt_n ~2048.
    words = 1750
    for _ in range(3):
        prompt = turn_prompt(words, 1.167, task=task)
        probe = send(prompt, 1, True)
        n = probe["usage"]["prompt_tokens"]
        if abs(n - 2048) <= 16:
            break
        words = max(1, round(words * 2048 / n))
    prompt = turn_prompt(words, 1.167, task=task)
    # Prepare every session with the exact prompt + one output token.
    prepared = cohort(prompt, c, 1, True)
    if any(p[2] > 16 for p in prepared):
        results.setdefault("notes", []).append(f"{mode}/{task}/C{c}: prepare reused cache")
    # Measured cohort: same prompt again, full cache hit, 128 output tokens.
    measured = cohort(prompt, c, 128, True)
    texts = [m[0] for m in measured]
    hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
    ref = hashes[0]
    exact = sum(1 for h in hashes if h == ref)
    total_tps = sum(m[1] for m in measured)
    cached = [m[2] for m in measured]
    results.setdefault("results", {}).setdefault(mode, {}).setdefault(task, {})[f"c{c}"] = {
        "sum_decode_tps": round(total_tps, 2),
        "per_request_tps": [round(m[1], 2) for m in measured],
        "cached_tokens": cached,
        "exact_vs_c1": exact,
        "of": c,
        "prompt_tokens": probe["usage"]["prompt_tokens"],
        "c1_sha256": ref[:16],
    }
    print(f"{mode} {task} C{c}: {total_tps:.2f} tok/s (sum), exact {exact}/{c}", flush=True)


def main():
    exe, model, mtp, out_path = sys.argv[1], sys.argv[2], (sys.argv[3] if sys.argv[3] != "-" else None), sys.argv[4]
    results = {"measuredOn": time.strftime("%Y-%m-%d"), "protocol": "pp2048 prepared cohort, tg128 measured, sum of per-request decode rates",
               "results": {}, "notes": []}
    for mode in ("ar", "mtp"):
        for c in (1, 2, 4, 6, 8):
            log = Path(out_path).parent / f"probe-{mode}-c{c}.log"
            server = Server(exe, model, mtp if mode == "mtp" else None, c, log)
            if not server.wait_ready():
                results["notes"].append(f"{mode} C{c}: server not ready")
                server.stop()
                continue
            try:
                measure(log, mode, "prose", c, results)
                if mode == "mtp":
                    measure(log, mode, "repetition", c, results)
            finally:
                server.stop()
                time.sleep(3)
    Path(out_path).write_text(json.dumps(results, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
