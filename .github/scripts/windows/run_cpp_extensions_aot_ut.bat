call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\Intel\oneAPI\compiler\latest\env\vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\ocloc\latest\env\vars.bat"
call "C:\Program Files (x86)\Intel\oneAPI\pti\latest\env\vars.bat"
call conda activate windows_ut
cd ../pytorch/third_party/torch-xpu-ops/test/xpu/
python -m pytest test_cpp_extensions_aot_xpu.py -v --junit-xml=test_cpp_extensions_aot_xpu.xml

if not exist "%GITHUB_WORKSPACE%\ut_log" mkdir "%GITHUB_WORKSPACE%\ut_log"
copy test_cpp_extensions_aot_xpu.xml %GITHUB_WORKSPACE%\ut_log /Y
