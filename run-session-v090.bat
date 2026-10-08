@echo off
rem Session test runner for the issue-6 worktree.
rem The test exe must be COPIED into the build root before running: its own
rem directory (tests\models\qwen38_flash_next) holds no HIP DLLs, so Windows
rem DLL search falls through to C:\Windows\System32\amdhip64_7.dll (the
rem driver's old runtime) whose comgr cannot find matching device bitcode —
rem instant segfault. From the build root the exe loads the staged 10.0.0
rem runtime. CWD is irrelevant; the exe location decides.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d D:\gufo-issue6\build\windows-release
copy /y tests\models\qwen38_flash_next\qwen38_flash_next_session_test.exe . >nul
set SNAP=C:/Users/EndyJasmi/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/38bb39ee97821de2c9009abb7e93950eec396e66
qwen38_flash_next_session_test.exe --model "%SNAP%/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf" --mtp-model "%SNAP%/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf" --sampling-only
set EC=%ERRORLEVEL%
del qwen38_flash_next_session_test.exe
if not "%EC%"=="0" exit /b %EC%
echo SAMPLING_SUITE_OK
