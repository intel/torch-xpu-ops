@echo off
REM Build PyTorch XPU wheel on Windows using PyTorch's upstream build scripts.
REM Assumes PyTorch source is already prepared (via prepare_pytorch.py) at
REM %WORKSPACE%\pytorch and oneAPI is already installed via xpu_install.bat.
REM
REM Environment variables (set by caller):
REM   WORKSPACE          - workspace root (parent of pytorch/)
REM   XPU_VERSION        - XPU bundle version (for xpu_install.bat)

setlocal enabledelayedexpansion
echo on

REM === Activate conda environment ===
call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call conda activate windows_ci

cd /d "%WORKSPACE%\pytorch"

REM === Run build via Python (avoids cmd quoting/path issues) ===
python "%WORKSPACE%\torch-xpu-ops\.github\scripts\build_windows_helper.py"
if errorlevel 1 exit /b 1

REM === Install and verify ===
for /r dist %%i in (torch*.whl) do (
    set TORCH_WHL=%%i
)
echo [INFO] Built wheel: !TORCH_WHL!
python -m pip install "!TORCH_WHL!"
if errorlevel 1 exit /b 1

REM === Verify XPU compilation ===
python -c "import torch; assert torch.xpu._is_compiled(), 'XPU not compiled!'"
if errorlevel 1 (
    echo Build verification failed!
    exit /b 1
)

echo [INFO] Build completed successfully
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
