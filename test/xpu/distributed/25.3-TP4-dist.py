[init] world_size=4, rank=0, xpu=0
[benchmark] start size=8 KB, warmup=20, loop=100, measure(last)=50
2026:06:11-20:53:48:3612983:[0] |CCL_WARN| topology recognition shows PCIe connection between devices. If this is not correct, you can disable topology recognition, with CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0. This will assume XeLinks across devices
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
| 8.00 KB | 94.73 | 12.44 | 122.35 | 109.91 | 0.130 | 12.44 | 0.988 |
| 16.00 KB | 35.20 | 8.23 | 53.85 | 45.62 | 0.698 | 8.23 | 2.987 |
| 32.00 KB | 26.94 | 8.63 | 46.55 | 37.92 | 1.825 | 8.63 | 5.692 |
| 64.00 KB | 53.35 | 9.34 | 90.18 | 80.84 | 1.843 | 9.34 | 10.526 |
| 128.00 KB | 124.05 | 11.88 | 244.65 | 232.77 | 1.585 | 11.88 | 16.551 |
| 256.00 KB | 131.63 | 20.87 | 192.32 | 171.45 | 2.987 | 20.87 | 18.841 |
| 512.00 KB | 132.43 | 39.88 | 182.01 | 142.13 | 5.938 | 39.88 | 19.717 |
| 1.00 MB | 181.90 | 81.76 | 261.33 | 179.57 | 8.647 | 81.76 | 19.238 |
| 2.00 MB | 245.21 | 154.80 | 468.22 | 313.42 | 12.829 | 154.80 | 20.321 |
| 4.00 MB | 308.43 | 304.25 | 316.28 | 12.03 | 20.398 | 304.25 | 20.678 |
| 8.00 MB | 648.38 | 609.32 | 704.09 | 94.77 | 19.407 | 609.32 | 20.651 |

[last-50-table] rows=rank, cols=iteration-id, value=kernel_us
| rank/iter | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 | 33 | 34 | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | 48 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 609.79 | 618.23 | 605.52 | 741.77 | 609.58 | 752.60 | 607.39 | 606.88 | 666.67 | 606.98 | 675.62 | 607.92 | 673.02 | 605.83 | 665.94 | 606.35 | 667.60 | 608.44 | 616.04 | 610.42 | 666.67 | 606.56 | 661.98 | 609.89 | 663.85 | 609.79 | 617.81 | 606.56 | 609.79 | 606.77 | 666.77 | 607.81 | 656.46 | 607.60 | 666.88 | 607.08 | 617.81 | 607.08 | 664.17 | 606.98 | 619.38 | 608.75 | 659.17 | 606.56 | 660.62 | 606.04 | 667.39 | 607.50 | 669.27 | 608.54 |
| 1 | 607.50 | 606.67 | 1274.89 | 673.64 | 677.81 | 675.94 | 667.71 | 676.77 | 673.33 | 667.19 | 674.79 | 673.33 | 622.08 | 625.21 | 670.62 | 667.60 | 669.58 | 674.17 | 620.21 | 674.38 | 675.42 | 619.69 | 640.00 | 620.00 | 633.64 | 667.81 | 621.67 | 664.69 | 670.62 | 667.71 | 664.06 | 620.52 | 663.85 | 666.14 | 667.60 | 618.85 | 621.67 | 667.39 | 619.48 | 670.10 | 669.06 | 667.19 | 666.56 | 674.17 | 618.96 | 673.23 | 669.27 | 663.64 | 667.19 | 668.85 |
| 2 | 606.67 | 649.27 | 606.46 | 618.23 | 606.14 | 618.54 | 647.50 | 607.92 | 613.85 | 610.52 | 607.92 | 650.52 | 607.39 | 648.64 | 644.06 | 605.52 | 615.94 | 606.04 | 620.21 | 607.50 | 652.08 | 608.44 | 606.67 | 733.64 | 607.81 | 838.44 | 606.67 | 664.38 | 608.75 | 662.81 | 606.56 | 666.67 | 607.29 | 660.62 | 607.60 | 617.81 | 609.89 | 661.88 | 609.27 | 668.44 | 609.27 | 617.81 | 608.85 | 665.73 | 606.77 | 610.31 | 606.67 | 620.52 | 607.81 | 613.33 |
| 3 | 619.17 | 672.39 | 615.73 | 635.83 | 661.88 | 619.79 | 615.21 | 666.46 | 616.88 | 636.77 | 613.96 | 632.29 | 666.88 | 662.29 | 615.31 | 620.10 | 620.83 | 661.77 | 627.19 | 661.25 | 665.21 | 665.52 | 617.29 | 624.27 | 640.10 | 618.96 | 663.33 | 744.48 | 607.08 | 753.33 | 607.08 | 1276.88 | 672.50 | 678.12 | 676.35 | 667.50 | 676.25 | 672.92 | 669.48 | 673.02 | 675.31 | 622.71 | 624.48 | 671.56 | 667.29 | 669.17 | 674.58 | 620.94 | 674.79 | 675.83 |
[done] destroy process group
