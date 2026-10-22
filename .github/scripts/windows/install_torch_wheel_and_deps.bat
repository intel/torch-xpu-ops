call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call conda activate windows_ut
set PYTORCH_INPUT=%PYTORCH_REF%
echo %PYTORCH_INPUT% | findstr /C:"_wheel" >nul
if %errorlevel% equ 0 (
  echo "Installing PyTorch from nightly wheel..."
  python -m pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/xpu
  REM Clone pytorch source at the installed version's commit for test infra
  for /f "delims=" %%v in ('python -c "import torch; print(torch.version.git_version)"') do set TORCH_COMMIT_ID=%%v
  echo "Installed torch commit: %TORCH_COMMIT_ID%"
  cd %GITHUB_WORKSPACE%\..
  git clone https://github.com/pytorch/pytorch.git pytorch
  cd pytorch && git checkout %TORCH_COMMIT_ID%
  REM Prepare torch-xpu-ops for test scripts
  set torch_xpu_ops=%TORCH_XPU_OPS_REF%
  if "%torch_xpu_ops%" == "pinned" (
    for /f "delims=" %%c in ('type third_party\xpu.txt') do set XPU_OPS_COMMIT=%%c
    cd third_party
    if exist "torch-xpu-ops" rmdir /s /q torch-xpu-ops
    cd ..
    git clone "https://github.com/intel/torch-xpu-ops.git" "third_party\torch-xpu-ops"
    cd "third_party\torch-xpu-ops"
    git checkout "%XPU_OPS_COMMIT%"
    cd ../..
  ) else (
    cd third_party
    if exist "torch-xpu-ops" rmdir /s /q torch-xpu-ops
    cd ..
    git clone "https://github.com/intel/torch-xpu-ops.git" "third_party\torch-xpu-ops"
    cd "third_party\torch-xpu-ops"
    git checkout "%torch_xpu_ops%"
    cd ../..
  )
) else (
  REM Install in the loop body: %TORCH_WHL% would expand when cmd
  REM parses this block, before the for has assigned it.
  for /r "%GITHUB_WORKSPACE%\wheel_artifact" %%f in (*.whl) do python -m pip install "%%f"
  python -c "import torch" || exit /b 1
)
cd %GITHUB_WORKSPACE%\..\pytorch
python -m pip install -r .ci\docker\requirements-ci.txt
