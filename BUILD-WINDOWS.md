# Building Gufo (Windows / Strix Halo) from my fork

This is the native Windows port of Gufo for AMD Strix Halo (Ryzen AI Max+ 395, gfx1151). It is **not** an upstream-supported configuration — upstream targets Linux — so the build expects a specific toolchain layout. Follow the steps in order.

## 1. Prerequisites

- **Hardware/driver:** AMD Strix Halo (gfx1151) with a recent AMD driver (current Adrenalin/PRO drivers work).
- **Visual Studio 2022/18** with the "Desktop development with C++" workload (MSVC x64 — the device compiler comes from TheRock below, not VS).
- **Git**, **CMake ≥ 3.29**, **Ninja** (VS bundles both under `Common7\IDE\CommonExtensions\Microsoft\CMake\bin`), **Python 3**.
- **vcpkg** cloned somewhere fixed, e.g. `C:\vcpkg`:

  ```cmd
  git clone https://github.com/microsoft/vcpkg.git C:\vcpkg && C:\vcpkg\bootstrap-vcpkg.bat
  ```

- **TheRock HIP SDK** (the AMD ROCm development distribution built by the TheRock project) — a **Windows nightly distribution for gfx1151**, extracted to exactly **`C:\TheRock\build`**. The build presets hardcode `C:/TheRock/build`, so if you put it elsewhere, also update the compiler/prefix paths in `CMakePresets.json` (`windows-base` preset). Use a **nightly** build — the stable ROCm 7.x SDK runtimes currently crash (`hipStreamCreateWithFlags` hangs/segfaults) with shipping drivers.

## 2. Patch the SDK's HIP headers (once, and after every SDK reinstall)

MSVC 14.40+ conflicts with Clang's HIP math headers. The repo ships an idempotent fixer:

```cmd
python tools\windows\patch_sdk_math.py C:\TheRock\build
```

## 3. Clone

```cmd
git clone -b feature/window-native https://github.com/endyjasmi/gufo.git
cd gufo
```

## 4. Fix the VS path in the env script

`tools\windows\vs-env.cmd` hardcodes `C:\Program Files\Microsoft Visual Studio\18\Community\...`. If your VS is installed elsewhere (e.g. `2022` instead of `18`, or a different edition), edit that one `call ... vcvars64.bat` line to point at your `vcvars64.bat`.

## 5. Configure + build

```cmd
tools\windows\vs-env.cmd cmake --preset windows-release -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
tools\windows\vs-env.cmd cmake --build --preset windows-release
```

(Adjust the vcpkg path to where you put it. First configure downloads dependencies via vcpkg. The build itself is a few minutes and produces `build\windows-release\gufo.exe` plus the staged ROCm DLLs.)

**The binary must stay in `build\windows-release\`** — it won't run from anywhere else.

## 6. Copy the Tensile data directories (required — the build does not do this)

The build stages the rocBLAS/hipBLASLt DLLs but **not** their kernel libraries. Without this step every GPU call fails with `rocBLAS error: Could not initialize Tensile host ... /rocblas/library`:

```cmd
xcopy /E /I C:\TheRock\build\bin\rocblas   build\windows-release\rocblas
xcopy /E /I C:\TheRock\build\bin\hipblaslt build\windows-release\hipblaslt
```

Re-do this whenever you rebuild into a fresh folder or reinstall the SDK.

## 7. Verify

```cmd
build\windows-release\gufo.exe diagnose
```

Then run a real model (GGUF from Hugging Face, e.g. an unsloth Q4_K_XL build):

```cmd
build\windows-release\gufo.exe bench -p 2048 -n 128 <path\to\model.gguf>
```

To serve:

```cmd
build\windows-release\gufo.exe serve --port 8080 llm --model <path\to\model.gguf> --context 4096
```

Then `curl http://127.0.0.1:8080/ready` and POST to `/v1/chat/completions`.

## Gotchas

- If linking fails with `failed to write output 'gufo.exe': permission denied`, a `gufo serve` process is still running — kill it and rebuild.
- Don't move/copy `gufo.exe` out of `build\windows-release\` (or if you do, take the whole folder: `gufo.exe`, all DLLs, and the `rocblas\` and `hipblaslt\` directories).
- If you reinstall or upgrade the TheRock SDK, re-run steps 2 and 6.
- Don't run repo formatting tools on Windows; the CI toolchain differs.

---

For reference: on a comparable machine the fresh build produced a **2.4 GB** `build\windows-release` folder and took about **4 minutes** after configure.
