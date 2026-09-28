#!/usr/bin/env python3
"""Windows launcher for tools/bench/model-bench.py.

The shared driver's server lifecycle assumes POSIX process groups; map the
two calls it makes onto Windows process termination. Everything else (HTTP,
timings, artifacts, rendering) is portable. The loading and memory tables
stay Linux-only (privileged page-cache drop, ldd-resolved libamdhip64) and
are simply not requested on Windows.

Usage: python tools/windows/model-bench-win.py <model-bench args...>
"""
import os
import signal
import sys
from pathlib import Path

if not hasattr(os, "killpg"):
    def _killpg(pid: int, sig: int) -> None:
        os.kill(pid, sig)  # TerminateProcess; benchmark servers hold no state worth a graceful stop

    os.killpg = _killpg
if not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gufo.model_bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
