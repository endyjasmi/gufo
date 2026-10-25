"""One quick real-model check per model family against an owned gufo server.

Standard library only, so a runner without pip packages (the Windows Devlab
host) can run it next to a downloaded gufo bundle:

    python tests/functional/family_smoke.py embedding \
        --binary gufo.exe --model <embeddinggemma-2 snapshot>

Each family starts `gufo serve`, waits for /ready, makes one request, checks
one property against a committed reference and stops the server. The broader
suites (embeddings.py, asr_service.py, ...) stay the quality gate.
"""

import argparse
import json
import math
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
MINIMUM_EMBEDDING_COSINE = 0.9999


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(url, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {} if payload is None else {"Content-Type": "application/json"}
    with urllib.request.urlopen(urllib.request.Request(url, data, headers),
                                timeout=timeout) as response:
        return json.load(response)


def wait_ready(url, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode} before /ready")
        try:
            request(url + "/ready", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f"server not ready after {timeout} s")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


def check_embedding(url, model):
    golden = json.loads((FIXTURES / "embedding_gemma2" / "embedding_golden.json")
                        .read_text(encoding="utf-8"))
    cases = golden["cases"]
    result = request(url + "/v1/embeddings", {
        "model": model, "input": [case["text"] for case in cases], "encoding_format": "float"})
    items = sorted(result["data"], key=lambda item: item["index"])
    assert len(items) == len(cases), f"{len(items)} embeddings for {len(cases)} inputs"
    worst = 1.0
    for case, item in zip(cases, items):
        vector = item["embedding"]
        assert len(vector) == len(case["embedding"]), f"{case['name']}: {len(vector)} dimensions"
        similarity = cosine(vector, case["embedding"])
        print(f"  {case['name']}: cosine {similarity:.6f}")
        worst = min(worst, similarity)
    assert worst >= MINIMUM_EMBEDDING_COSINE, f"worst cosine {worst:.6f} < {MINIMUM_EMBEDDING_COSINE}"
    return f"{len(cases)} golden texts, worst cosine {worst:.6f}"


FAMILIES = {
    "embedding": ("embedding", "embeddinggemma-2", check_embedding),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("family", choices=sorted(FAMILIES))
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--log", type=Path, help="Server log (default: <family>-server.log)")
    args = parser.parse_args()

    service, public_model, check = FAMILIES[args.family]
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    command = [str(args.binary), "serve", service, "--model", str(args.model),
               "--served-model-name", public_model, "--host", "127.0.0.1", "--port", str(port)]
    log_path = args.log or Path(f"{args.family}-server.log")
    print("Starting:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            started = time.monotonic()
            wait_ready(url, process, args.startup_timeout)
            print(f"ready in {time.monotonic() - started:.1f} s", flush=True)
            started = time.monotonic()
            summary = check(url, public_model)
            print(f"PASS {args.family}: {summary} ({time.monotonic() - started:.1f} s)")
        except Exception as error:
            print(f"FAIL {args.family}: {error}", file=sys.stderr)
            log.flush()
            print(log_path.read_text(encoding="utf-8", errors="replace")[-4000:], file=sys.stderr)
            return 1
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
