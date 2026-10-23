set XPU_VERSION=%XPU_VERSION_INPUT%
call .github\scripts\install_xpu.bat
if errorlevel 1 exit /b 1
