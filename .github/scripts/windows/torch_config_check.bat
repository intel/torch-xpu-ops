call "C:\ProgramData\miniforge3\Scripts\activate.bat"
call conda activate windows_ut
python -c "import torch; print(torch.__config__.show())"
python -c "import torch; print(torch.__config__.parallel_info())"
python -c "import torch; print(torch.xpu.device_count())"
