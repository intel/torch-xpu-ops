call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call conda activate windows_ut
cd ../pytorch/third_party/torch-xpu-ops/test/xpu/
python run_test_win_with_skip_mtl.py

if not exist "%GITHUB_WORKSPACE%\ut_log" mkdir "%GITHUB_WORKSPACE%\ut_log"
copy test_xpu.xml %GITHUB_WORKSPACE%\ut_log /Y
