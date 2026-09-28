@echo off
rem Stable ROCm 7.1 SDK (extracted to C:\HIPSDK7\rocm-7.1) environment for
rem gufo's experimental side-by-side build. Shadows the machine-wide HIP_PATH
rem which points at the TheRock (ROCm 10 nightly) dist; that dist is untouched.
set "HIP_PATH=C:\HIPSDK7\rocm-7.1"
set "HIP_DEVICE_LIB_PATH=C:\HIPSDK7\rocm-7.1\amdgcn\bitcode"
set "HIP_PLATFORM=amd"
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\HIPSDK7\rocm-7.1\bin;%PATH%"
set "PATH=C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PATH%"
%*
