call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call conda activate windows_ci
set WORKSPACE=%GITHUB_WORKSPACE%\..
set pytorch=%PYTORCH_REF%
set torch_xpu_ops=%TORCH_XPU_OPS_REF%

REM Git config for long paths and symlinks (required for pytorch submodules)
git config --system core.longpaths true
git config --global core.symlinks true
git config --global core.fsmonitor false
powershell -Command "Set-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name 'LongPathsEnabled' -Value 1"

REM Parse pytorch repo/commit
echo %pytorch% | findstr /C:"https://" >nul
if %errorlevel% equ 0 (
    echo %pytorch% | findstr "@" >nul
    if %errorlevel% equ 0 (
        for /f "tokens=1,2 delims=@" %%a in ("%pytorch%") do (
            set PYTORCH_REPO=%%a
            set PYTORCH_COMMIT=%%b
        )
    ) else (
        set PYTORCH_REPO=%pytorch%
        set PYTORCH_COMMIT=main
    )
) else (
    set PYTORCH_REPO=https://github.com/pytorch/pytorch.git
    set PYTORCH_COMMIT=%pytorch%
)

REM Parse torch-xpu-ops repo/commit
echo %torch_xpu_ops% | findstr /C:"https://" >nul
if %errorlevel% equ 0 (
    echo %torch_xpu_ops% | findstr "@" >nul
    if %errorlevel% equ 0 (
        for /f "tokens=1,2 delims=@" %%a in ("%torch_xpu_ops%") do (
            set TORCH_XPU_OPS_REPO=%%a
            set TORCH_XPU_OPS_COMMIT=%%b
        )
    ) else (
        set TORCH_XPU_OPS_REPO=%torch_xpu_ops%
        set TORCH_XPU_OPS_COMMIT=main
    )
) else (
    set TORCH_XPU_OPS_REPO=https://github.com/intel/torch-xpu-ops.git
    set TORCH_XPU_OPS_COMMIT=%torch_xpu_ops%
)

python .github\scripts\prepare_pytorch.py --workspace=%WORKSPACE% --pytorch-repo=%PYTORCH_REPO% --pytorch-commit=%PYTORCH_COMMIT% --torch-xpu-ops-repo=%TORCH_XPU_OPS_REPO% --torch-xpu-ops-commit=%TORCH_XPU_OPS_COMMIT%
if errorlevel 1 exit /b 1
