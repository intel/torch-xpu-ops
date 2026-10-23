call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call conda activate windows_ci
set WORKSPACE=%GITHUB_WORKSPACE%\..
python .github\scripts\build_windows.py > %WORKSPACE%\pytorch\build_torch_wheel_log.log 2>&1
set BUILD_ERR=%ERRORLEVEL%
type %WORKSPACE%\pytorch\build_torch_wheel_log.log
if %BUILD_ERR% neq 0 exit /b 1
cd ..\pytorch
for /r %%i in (dist\torch*.whl) do (
  pip install --force-reinstall "%%i"
)
cd %TEMP%
python -c "import torch; assert torch.xpu._is_compiled(), 'XPU not compiled!'"
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
