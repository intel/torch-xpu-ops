call "C:\ProgramData\miniforge3\Scripts\activate.bat"
if exist "C:\ProgramData\miniforge3\envs\windows_ci" (
    rmdir /s /q "C:\ProgramData\miniforge3\envs\windows_ci"
)
call conda clean -ay
call conda remove --all -y -n windows_ci
call conda create -n windows_ci python=%PYTHON_VERSION% cmake ninja -y
