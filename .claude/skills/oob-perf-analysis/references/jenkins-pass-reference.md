# Jenkins Pass Reference

The OOB eager workflow uses six Jenkins passes.

| Pass | Device | Purpose |
|------|--------|---------|
| `t1` | CUDA | FLOPs and bytes projection input |
| `unitrace` | XPU | Kernel-level GPU timing on XPU |
| `xpu_profiler` | XPU | Per-op GPU timing trace on XPU |
| `cuda_profiler` | CUDA | Per-op GPU timing trace on CUDA |
| `xpu_t2` | XPU | XPU wall-clock batch latency |
| `cuda_t2` | CUDA | CUDA wall-clock batch latency |

These pass definitions are the source of truth for session workflow planning.
