@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d D:\gufo-issue6
"C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build\windows-release --target qwen38_flash_next_gpu_probe --parallel 8
if errorlevel 1 exit /b 1
echo AUDIT_BUILD_OK
