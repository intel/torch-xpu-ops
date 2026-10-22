echo "C:\ProgramData\miniforge3\Scripts" >> "$GITHUB_PATH"
echo "C:\ProgramData\miniforge3\Library\bin" >> "$GITHUB_PATH"
call "C:\ProgramData\miniforge3\Scripts\activate.bat"
rmdir /s /q "C:\ProgramData\miniforge3\envs\windows_ut"
call conda clean -ay
call conda remove --all -y -n windows_ut
call conda create -n windows_ut python=%PYTHON_VERSION% -y
call conda activate windows_ut
python -m pip install pyyaml requests pytest-timeout pytest pytest-xdist
call conda install -y libuv
call conda install -y rust
cd ..
if exist "pytorch" (
  rmdir /s /q pytorch
)
set pytorch=%PYTORCH_REF%
set torch_xpu_ops=%TORCH_XPU_OPS_REF%

REM Skip pytorch clone for nightly wheel - will be handled after install
echo %pytorch% | findstr /C:"_wheel" >nul
if %errorlevel% equ 0 (
  echo "Nightly wheel mode: skipping pytorch clone (will clone after install)"
  goto :setup_done
)

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
git clone %PYTORCH_REPO% pytorch
cd pytorch && git checkout %PYTORCH_COMMIT%
if %TORCH_XPU_OPS_COMMIT% == 'pinned' (
  echo "Don't replace torch-xpu-ops!"
) else (
  echo "Replace torch-xpu-ops!"
  cd third_party
  if exist "torch-xpu-ops" (
    rmdir /s /q torch-xpu-ops
  )
  cd ..
  if "%GITHUB_EVENT_NAME%" == "pull_request" (
    Xcopy ..\torch-xpu-ops third_party\torch-xpu-ops /E/H/Y/F/I
  ) else (
    git clone "%TORCH_XPU_OPS_REPO%" "third_party\torch-xpu-ops"
    cd "third_party\torch-xpu-ops"
    git checkout "%TORCH_XPU_OPS_COMMIT%"
    cd ../..
  )
  powershell -Command "(Get-Content caffe2/CMakeLists.txt) -replace 'checkout --quiet \${TORCH_XPU_OPS_COMMIT}', 'log -n 1' | Set-Content caffe2/CMakeLists.txt"
)
:setup_done
