"""File the issue-6 follow-up issues on endyjasmi/gufo and post the findings comment."""
import json
import subprocess
import urllib.request

REPO = "endyjasmi/gufo"
API = f"https://api.github.com/repos/{REPO}"


def token():
    out = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n",
        capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    raise SystemExit("no stored credential")


def api(tok, method, path, payload=None):
    req = urllib.request.Request(
        f"{API}/{path}", method=method,
        data=json.dumps(payload).encode("utf-8") if payload else None,
        headers={"Authorization": f"token {tok}",
                 "Content-Type": "application/json",
                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


tok = token()

with open("triage/issue-depth-gate.md", encoding="utf-8") as f:
    depth_body = f.read()
with open("triage/issue-interleave.md", encoding="utf-8") as f:
    interleave_body = f.read()

a = api(tok, "POST", "issues", {
    "title": "qwen38-flash-next: verify-cycle logits diverge from width-1 "
             "decode at context \u2265 2048 (row-count-dependent sparse "
             "attention/indexer)",
    "body": depth_body, "labels": ["bug"]})
print("created A:", a["number"], a["html_url"])

b = api(tok, "POST", "issues", {
    "title": "session_test --sampling-only: interleaved-pair check fails on "
             "pristine Windows builds",
    "body": interleave_body, "labels": ["bug"]})
print("created B:", b["number"], b["html_url"])

comment = f"""Update from the fix + v0.9.0 re-validation (branch `triage/issue-6-empty-turns` @ `8730d256`, full write-up in `triage/TRIAGE.md`):

**Landed:** per-row PLE injection + per-row head mixer (batched vocabulary projection) at decode sizes — decode-sized batches no longer compute routing-relevant logits differently from single-token decode. Re-validated on the v0.9.0 base: all serial sampling gates pass, and a fresh 8-session arm of the fixture measured 9/56 empty turns (pooled with the earlier arms 15/126, 11.9%) vs 12/56 (21.4%) before the fix. The AR-only reference stays 1/56 (1.8%).

**Not fully solved — and now precisely characterized:** the residual degeneration is depth-gated. Within the dense-attention region (< 2048 tokens = `indexer_top_k`) MTP verify is bit-exact vs AR; at the fixture's working depths (2.4–4K tokens) the sparse attention/indexer decode path is row-count-dependent, so every verify cycle re-rolls near-ties and MTP's empty-turn risk stays several times the AR-only rate. The deterministic reproducer and the acceptance bar for the kernel work are in #{a['number']}. Until that lands, `--speculative off` remains the clean workaround (0/56 across two full runs here).

Separately re-confirmed on the v0.9.0 base: the `--sampling-only` interleaved-pair failure is pre-existing and unrelated — filed as #{b['number']}.
"""

c = api(tok, "POST", "issues/6/comments", {"body": comment})
print("commented on #6:", c["html_url"])
