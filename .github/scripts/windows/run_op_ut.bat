call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call conda activate windows_ut
set PYTORCH_TEST_WITH_SLOW=1
set PYTEST_ADDOPTS=-v --timeout 600 --timeout_method=thread --max-worker-restart 1000000 -n 1
cd ../pytorch/third_party/torch-xpu-ops/test/xpu/
python run_test_with_windows_nightly.py
if not exist "%GITHUB_WORKSPACE%\ut_log" mkdir "%GITHUB_WORKSPACE%\ut_log"
for /r . %%f in (test*.xml op_ut_with_*.xml) do move "%%f" "%GITHUB_WORKSPACE%\ut_log\" >nul 2>&1 || echo "File move completed"
