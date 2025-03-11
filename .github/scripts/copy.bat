copy "%CONDA_PREFIX%\Library\bin\libiomp*5md.dll" .\torch\lib
:: Should be set in build_pytorch.bat
copy "%CONDA_PREFIX%\Library\bin\uv.dll" .\torch\lib

if defined CMAKE_PREFIX_PATH (
    set CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library;%CMAKE_PREFIX_PATH%
) else (
    set CMAKE_PREFIX_PATH=%CONDA_PREFIX%\Library
)