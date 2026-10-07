@echo off
rem Issue #6 fix build: worktree D:\gufo-issue6, TheRock 10.0.0 toolchain,
rem same flags as the control recipe (D:\gufo-control\configure-control.bat)
rem plus BUILD_TESTING for the parity probe.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
set CMAKE="C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
cd /d D:\gufo-issue6
%CMAKE% -S . -B build\windows-release -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=C:/TheRock/build/lib/llvm/bin/clang-cl.exe -DCMAKE_CXX_COMPILER=C:/TheRock/build/lib/llvm/bin/clang-cl.exe -DCMAKE_HIP_COMPILER=C:/TheRock/build/lib/llvm/bin/clang-cl.exe -DCMAKE_HIP_PLATFORM=amd -DHIP_PLATFORM=amd -DHIPCUB_INCLUDE_DIR=C:/TheRock/build/include -DROCPRIM_INCLUDE_DIR=C:/TheRock/build/include -DROCWMMA_INCLUDE_DIR=C:/TheRock/build/include -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows -DBUILD_TESTING=ON -DGUFO_BUILD_TOOLS=OFF -DGUFO_ENABLE_SANITIZERS=OFF -DGUFO_ENABLE_WARNINGS=ON -DGUFO_FFMPEG_EXECUTABLE=ffmpeg -DGUFO_FFPROBE_EXECUTABLE=ffprobe
if errorlevel 1 exit /b 1
%CMAKE% --build build/windows-release --target gufo --parallel 8
if errorlevel 1 exit /b 1
echo ISSUE6_GUFO_OK
