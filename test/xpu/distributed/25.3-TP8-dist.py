[init] world_size=8, rank=0, xpu=0
[benchmark] start size=8 KB, warmup=20, loop=100, measure(last)=50
2026:06:11-20:54:10:3614083:[0] |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume XeLinks across devices
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
  return func(*args, **kwargs)
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/profiler/profiler.py:224: UserWarning: Warning: Profiler clears events at the end of each cycle.Only events from the current cycle will be reported.To keep events across cycles, set acc_events=True.
  _warn_once(
/root/miniforge3/envs/hanchao/lib/python3.12/site-packages/torch/distributed/c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You can specify `device_id` in `init_process_group` to mute this warning.
  return func(*args, **kwargs)
[benchmark] done size=8 KB
[benchmark] start size=16 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=16 KB
[benchmark] start size=32 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=32 KB
[benchmark] start size=64 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=64 KB
[benchmark] start size=128 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=128 KB
[benchmark] start size=256 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=256 KB
[benchmark] start size=512 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=512 KB
[benchmark] start size=1024 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=1024 KB
[benchmark] start size=2048 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=2048 KB
[benchmark] start size=4096 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=4096 KB
[benchmark] start size=8192 KB, warmup=20, loop=100, measure(last)=50
[benchmark] done size=8192 KB

[summary-table] allreduce trace metrics
| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) | minAvg(last50)_us | busBW_minAvg(GB/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8.00 KB | 110.55 | 26.04 | 123.01 | 96.97 | 0.130 | 26.04 | 0.551 |
| 16.00 KB | 82.55 | 23.63 | 137.60 | 113.97 | 0.347 | 23.63 | 1.213 |
| 32.00 KB | 131.94 | 18.16 | 358.22 | 340.06 | 0.435 | 18.16 | 3.158 |
| 64.00 KB | 119.94 | 20.07 | 181.75 | 161.68 | 0.956 | 20.07 | 5.715 |
| 128.00 KB | 167.53 | 23.25 | 285.34 | 262.09 | 1.369 | 23.25 | 9.864 |
| 256.00 KB | 152.48 | 35.82 | 244.24 | 208.42 | 3.009 | 35.82 | 12.808 |
| 512.00 KB | 198.47 | 75.30 | 284.06 | 208.76 | 4.623 | 75.30 | 12.184 |
| 1.00 MB | 258.31 | 140.80 | 452.95 | 312.15 | 7.104 | 140.80 | 13.033 |
| 2.00 MB | 325.49 | 240.65 | 444.55 | 203.90 | 11.275 | 240.65 | 15.250 |
| 4.00 MB | 473.51 | 398.06 | 636.03 | 237.96 | 15.501 | 398.06 | 18.439 |
| 8.00 MB | 753.31 | 710.34 | 890.97 | 180.63 | 19.487 | 710.34 | 20.666 |

[last-50-table] rows=rank, cols=iteration-id, value=kernel_us
| rank/iter | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 708.12 | 704.58 | 710.83 | 705.00 | 710.42 | 706.67 | 707.19 | 707.81 | 706.14 | 710.00 | 706.56 | 709.69 | 705.62 | 708.54 | 707.08 | 705.83 | 710.62 | 707.60 | 1162.60 | 972.60 | 1105.10 | 1001.56 | 1005.00 | 999.48 | 989.89 | 946.77 | 990.62 | 998.33 | 991.77 | 990.10 | 922.39 | 906.14 | 910.62 | 1076.25 | 910.10 | 863.54 | 905.42 | 920.31 | 903.54 | 908.02 | 899.27 | 909.48 | 992.81 | 882.29 | 827.39 | 805.73 | 819.38 | 779.79 | 833.96 | 830.73 |
| 1 | 765.52 | 763.02 | 711.77 | 766.35 | 709.06 | 712.60 | 712.60 | 719.89 | 709.58 | 765.83 | 715.31 | 709.69 | 709.38 | 708.54 | 712.39 | 718.96 | 840.00 | 709.89 | 709.58 | 706.88 | 709.38 | 707.08 | 706.98 | 709.17 | 705.00 | 714.69 | 708.96 | 709.38 | 717.08 | 709.17 | 707.08 | 773.02 | 709.69 | 705.94 | 711.25 | 705.73 | 710.83 | 707.29 | 706.25 | 707.50 | 707.81 | 709.69 | 705.52 | 717.92 | 708.54 | 823.96 | 707.08 | 706.25 | 710.94 | 705.42 |
| 2 | 766.14 | 758.64 | 716.35 | 766.14 | 709.69 | 711.88 | 713.54 | 719.69 | 709.79 | 765.62 | 715.31 | 708.44 | 710.42 | 708.64 | 710.31 | 721.04 | 839.89 | 709.48 | 707.39 | 709.27 | 708.44 | 706.14 | 708.23 | 708.33 | 706.25 | 715.10 | 709.06 | 709.38 | 717.60 | 708.44 | 706.25 | 773.85 | 709.17 | 706.04 | 710.62 | 706.04 | 710.73 | 705.42 | 707.29 | 706.98 | 708.23 | 709.06 | 705.52 | 718.54 | 707.81 | 823.75 | 707.19 | 707.29 | 710.00 | 706.14 |
| 3 | 766.88 | 758.54 | 715.73 | 766.04 | 711.77 | 712.39 | 712.39 | 719.89 | 710.62 | 766.77 | 715.52 | 708.23 | 709.69 | 710.42 | 709.58 | 722.71 | 840.21 | 709.38 | 707.19 | 710.21 | 706.77 | 708.75 | 708.12 | 708.64 | 706.88 | 715.42 | 708.54 | 710.00 | 717.81 | 708.54 | 706.67 | 774.48 | 708.44 | 708.02 | 709.06 | 706.77 | 710.83 | 706.56 | 707.29 | 707.92 | 707.50 | 706.25 | 708.96 | 719.06 | 707.81 | 821.77 | 710.42 | 707.08 | 707.81 | 708.12 |
| 4 | 717.19 | 706.14 | 707.71 | 774.58 | 705.62 | 710.21 | 705.31 | 709.48 | 705.94 | 709.79 | 706.35 | 706.46 | 709.38 | 705.73 | 707.81 | 718.85 | 707.08 | 821.04 | 708.96 | 706.67 | 706.77 | 708.44 | 1193.02 | 1144.27 | 944.89 | 1001.46 | 1006.14 | 993.23 | 997.60 | 1105.62 | 991.67 | 997.50 | 994.79 | 990.94 | 922.08 | 903.12 | 904.17 | 857.08 | 909.79 | 1004.58 | 907.71 | 918.54 | 906.88 | 902.50 | 905.73 | 903.02 | 853.64 | 876.77 | 832.29 | 825.42 |
| 5 | 767.71 | 761.56 | 710.42 | 768.02 | 708.44 | 715.73 | 709.27 | 720.00 | 710.73 | 769.06 | 708.96 | 714.48 | 708.64 | 710.52 | 710.00 | 723.12 | 839.38 | 706.88 | 710.31 | 710.42 | 706.04 | 708.96 | 708.12 | 707.71 | 709.17 | 714.79 | 708.64 | 709.27 | 717.29 | 705.83 | 708.64 | 775.83 | 705.42 | 709.79 | 707.19 | 709.38 | 707.92 | 709.48 | 707.19 | 707.08 | 709.27 | 706.46 | 708.54 | 718.96 | 707.60 | 821.98 | 709.38 | 706.88 | 706.35 | 710.73 |
| 6 | 768.54 | 760.52 | 709.79 | 766.98 | 709.38 | 715.31 | 708.23 | 722.08 | 711.14 | 768.75 | 708.64 | 713.23 | 708.75 | 708.23 | 711.67 | 718.44 | 838.33 | 711.14 | 710.10 | 709.58 | 706.56 | 707.81 | 707.92 | 707.29 | 708.75 | 714.89 | 707.39 | 707.92 | 721.46 | 705.21 | 708.12 | 775.94 | 706.25 | 708.33 | 708.64 | 708.44 | 708.54 | 709.17 | 707.39 | 706.88 | 707.08 | 708.64 | 708.33 | 718.44 | 707.19 | 822.39 | 708.33 | 707.71 | 707.92 | 709.89 |
| 7 | 763.12 | 768.02 | 709.58 | 767.71 | 709.06 | 714.58 | 708.23 | 722.60 | 710.00 | 768.96 | 711.56 | 710.52 | 709.27 | 707.50 | 712.81 | 718.44 | 839.69 | 710.62 | 709.89 | 707.39 | 708.44 | 708.23 | 706.67 | 708.44 | 708.85 | 713.64 | 709.48 | 707.81 | 716.77 | 710.83 | 707.81 | 775.52 | 705.73 | 707.71 | 708.96 | 708.12 | 707.92 | 707.60 | 708.64 | 707.19 | 706.46 | 709.06 | 707.71 | 719.27 | 706.56 | 823.44 | 707.19 | 707.50 | 709.17 | 707.50 |
[done] destroy process group
