#!/usr/bin/env python3
"""Pack and run the weight-free gfx1151 tests for the Windows Devlab check.

A hosted Windows runner builds the tests and packs them, with their DLLs, the
HIP runtime and hipInfo, into one directory. The Devlab GPU runner, which sits
behind a slow link, downloads only that directory and runs it with the
standard library alone.

  select   list the weight-free GPU tests of a configured build (ctest JSON)
  collect  copy the selected executables and their runtime into a bundle
  run      run a bundle's tests and write a summary
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Tests that need model weights carry this label; the slow link rules them out.
EXTERNAL = 'external-model'

# The pull-request smoke: one fast weight-free test per class (about 7 s on
# Linux). Release PRs run every weight-free GPU test instead.
SMOKE = {
    'gufo_gpu_probe',                       # platform: device probe
    'gfx1151_acceptance_test',              # platform: gfx1151 features
    'hip_weight_upload_test',               # runtime: weight upload
    'hipblaslt_runtime_cache_test',         # runtime: hipBLASLt
    'persistent_deltanet_test',             # kernels: speculative DeltaNet
    'qwen38_flash_next.select_ops',         # Qwen3.8 Flash-Next
    'qwen_sampling_hip_test',               # Qwen 27B
    'ds4.projections',                      # DeepSeek V4 Flash
    'embedding_gemma2_attention_hip_test',  # EmbeddingGemma 2
    'minimax_h3_sampling_hip_test',         # MiniMax H3
    'qwen_image_21_kernel_test',            # Qwen-Image 2.1
}


def properties(test):
    return {item['name']: item['value'] for item in test.get('properties', [])}


def select(build, tier):
    shown = subprocess.run(['ctest', '--show-only=json-v1', '-L', 'gpu'], cwd=build,
                           check=True, capture_output=True, text=True).stdout
    roots = [str(build.resolve()), str(Path(__file__).resolve().parents[2])]
    selected = []
    for test in json.loads(shown)['tests']:
        command = test.get('command') or []
        props = properties(test)
        labels = set(props.get('LABELS', []))
        if not command or EXTERNAL in labels or props.get('ENVIRONMENT'):
            continue
        if tier == 'smoke' and test['name'] not in SMOKE:
            continue
        # Arguments naming source or build files would not exist in the bundle.
        if any(any(root.lower() in arg.lower() for root in roots) for arg in command[1:]):
            continue
        if Path(command[0]).suffix.lower() not in ('.exe', ''):
            continue  # Python-driven tests need the source tree.
        selected.append({'name': test['name'], 'exe': Path(command[0]).name,
                         'args': command[1:], 'path': command[0],
                         'skip_return_code': props.get('SKIP_RETURN_CODE'),
                         'timeout': props.get('TIMEOUT') or 600})
    return selected


def collect(tests, hip_path, extra, output):
    output.mkdir(parents=True, exist_ok=True)
    directories = set()
    for test in tests:
        exe = Path(test['path'])
        if not exe.suffix:
            exe = exe.with_suffix('.exe')
        shutil.copy2(exe, output / exe.name)
        test['exe'] = exe.name
        directories.add(exe.parent)
    for path in extra:
        shutil.copy2(path, output / path.name)
        directories.add(path.parent)
    # vcpkg copies its DLLs beside each executable; the HIP runtime and the
    # rocBLAS/hipBLASLt kernel data come from the SDK.
    for directory in sorted(directories):
        for dll in directory.glob('*.dll'):
            shutil.copy2(dll, output / dll.name)
    runtime = hip_path / 'bin'
    for dll in runtime.glob('*.dll'):
        shutil.copy2(dll, output / dll.name)
    for data in ('rocblas', 'hipblaslt'):
        if (runtime / data).is_dir():
            shutil.copytree(runtime / data, output / data, dirs_exist_ok=True)
    shutil.copy2(runtime / 'hipInfo.exe', output / 'hipInfo.exe')
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    for test in tests:
        test.pop('path', None)
    (output / 'tests.json').write_text(json.dumps({'tests': tests}, indent=2) + '\n')


def run(bundle, summary_path):
    bundle = bundle.resolve()
    tests = json.loads((bundle / 'tests.json').read_text())['tests']
    results = []
    for test in tests:
        started = time.monotonic()
        try:
            process = subprocess.run([str(bundle / test['exe']), *test['args']], cwd=bundle,
                                     capture_output=True, text=True, errors='replace',
                                     timeout=float(test['timeout']))
            code, output = process.returncode, process.stdout + process.stderr
        except subprocess.TimeoutExpired as error:
            code, output = None, f'timed out after {error.timeout} s'
        except OSError as error:
            code, output = None, f'cannot start: {error}'
        skip = test.get('skip_return_code')
        status = ('passed' if code == 0 else
                  'skipped' if skip is not None and code == int(skip) else 'failed')
        seconds = time.monotonic() - started
        results.append({'name': test['name'], 'status': status, 'exit_code': code,
                        'seconds': round(seconds, 1), 'tail': output[-4000:]})
        print(f'{status.upper():7} {test["name"]} ({seconds:.1f} s)', flush=True)
        if status == 'failed':
            print(output[-4000:], flush=True)
    failed = [r for r in results if r['status'] == 'failed']
    summary = {'status': 'failed' if failed or not results else 'passed', 'tests': results}
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    counts = {s: sum(r['status'] == s for r in results) for s in ('passed', 'skipped', 'failed')}
    print(f"{counts['passed']} passed, {counts['skipped']} skipped, {counts['failed']} failed")
    step_summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if step_summary:
        with open(step_summary, 'a') as out:
            out.write(f"\n### GPU tests: {summary['status']}\n\n"
                      f"{counts['passed']} passed, {counts['skipped']} skipped, {counts['failed']} failed\n\n"
                      "| Test | Status | Seconds |\n|---|---|---|\n")
            for r in results:
                out.write(f"| {r['name']} | {r['status']} | {r['seconds']} |\n")
    return 0 if summary['status'] == 'passed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    chooser = commands.add_parser('select')
    chooser.add_argument('--build', type=Path, required=True)
    chooser.add_argument('--tier', choices=('smoke', 'full'), default='smoke')
    chooser.add_argument('--output', type=Path, required=True)
    packer = commands.add_parser('collect')
    packer.add_argument('--tests', type=Path, required=True)
    packer.add_argument('--hip-path', type=Path, required=True)
    packer.add_argument('--extra', type=Path, action='append', default=[],
                        help='further executables to ship, e.g. gufo.exe')
    packer.add_argument('--output', type=Path, required=True)
    runner = commands.add_parser('run')
    runner.add_argument('--bundle', type=Path, required=True)
    runner.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'select':
        tests = select(args.build, args.tier)
        args.output.write_text(json.dumps(tests, indent=2) + '\n')
        # The build step compiles only these targets.
        print(' '.join(sorted({Path(t['exe']).stem for t in tests})))
        return 0
    if args.command == 'collect':
        collect(json.loads(args.tests.read_text()), args.hip_path, args.extra, args.output)
        return 0
    return run(args.bundle, args.summary)


if __name__ == '__main__':
    sys.exit(main())
