"""
Performance comparison: allreduce implementations vs torch.distributed.all_reduce

Usage:
    mpirun -n 2 python test_allreduce.py
    mpirun -n 2 python test_allreduce.py --impl symm_mem
    mpirun -n 2 python test_allreduce.py --impl pull
    mpirun -n 8 python test_allreduce.py --impl cross_switch
    mpirun -n 2 python test_allreduce.py --profile --impl symm_mem
    mpirun -n 2 python test_allreduce.py --accuracy --impl pull
"""

import argparse
import time
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
import os

from allreduce_impl import allreduce_with_symm_mem, hierarchical_allreduce_with_symm_mem, allreduce_cross_switch, allreduce_with_pull

# Map implementation names to functions
IMPL_MAP = {
    "symm_mem": allreduce_with_symm_mem,
    "pull": allreduce_with_pull,
    "cross_switch": allreduce_cross_switch,
}

def get_impl_func(impl_name: str):
    """Get the allreduce implementation function by name."""
    if impl_name not in IMPL_MAP:
        raise ValueError(f"Unknown implementation: {impl_name}. Available: {list(IMPL_MAP.keys())}")
    return IMPL_MAP[impl_name]


def init_distributed():
    """Initialize distributed environment."""
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29513'
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)

    return rank, world_size


def check_accuracy(tensor_size, rank, device, impl_func, dtype=torch.float32):
    """Detailed accuracy check for the specified allreduce implementation."""
    world_size = dist.get_world_size()
    tensor_size = (tensor_size // world_size) * world_size

    # Use same seed across ranks for reproducibility, but different data
    torch.manual_seed(42 + rank)
    tensor_ref = torch.randn(tensor_size, device=device, dtype=dtype)
    tensor_test = tensor_ref.clone()

    # Reference: torch.distributed.all_reduce
    dist.all_reduce(tensor_ref, op=dist.ReduceOp.SUM)

    # Test: specified implementation
    impl_func(tensor_test, op="sum")

    torch.xpu.synchronize()
    # Compute metrics
    abs_diff = torch.abs(tensor_ref - tensor_test)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    rel_diff = abs_diff / (torch.abs(tensor_ref) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    is_close = torch.allclose(tensor_ref, tensor_test, rtol=1e-5, atol=1e-5)

    return {
        "is_close": is_close,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
    }


def run_accuracy_check(impl_name: str):
    """Run detailed accuracy check."""
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    impl_func = get_impl_func(impl_name)

    sizes = [1048576, 4194304, 8388608, 33554432]

    if rank == 0:
        print("=" * 80)
        print(f"Accuracy Check: {impl_name} (world_size={world_size})")
        print("=" * 80)
        print(f"{'Size (MB)':>12} | {'Pass':>6} | {'MaxAbsDiff':>12} | {'MeanAbsDiff':>12} | {'MaxRelDiff':>12}")
        print("-" * 80)

    for size in sizes:
        metrics = check_accuracy(size, rank, device, impl_func)

        if rank == 0:
            status = "✓" if metrics["is_close"] else "✗"
            size_mb = size * 4 / (1024 * 1024)  # float32 = 4 bytes per element
            print(f"{size_mb:>12.1f} | {status:>6} | {metrics['max_abs_diff']:>12.2e} | "
                  f"{metrics['mean_abs_diff']:>12.2e} | {metrics['max_rel_diff']:>12.2e}")

    if rank == 0:
        print("=" * 80)


def run_profiler(impl_name: str, dtype=torch.bfloat16):
    """Run with torch profiler. Only rank 0 generates JSON trace file."""
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    impl_func = get_impl_func(impl_name)

    tensor_size = 8388608  # 16MB with BF16 (8M elements * 2 bytes)
    tensor_size = (tensor_size // world_size) * world_size
    num_iters = 10

    if rank == 0:
        print(f"Profiling {impl_name} (size={tensor_size}, dtype={dtype}, world_size={world_size})")

    # Warmup both implementations
    for _ in range(5):
        t = torch.randn(tensor_size, device=device, dtype=dtype)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = torch.randn(tensor_size, device=device, dtype=dtype)
        impl_func(t, op="sum")
    torch.xpu.synchronize()

    # Pre-allocate tensors outside profiling loop
    tensor_dist = torch.randn(tensor_size, device=device, dtype=dtype)
    tensor_impl = torch.randn(tensor_size, device=device, dtype=dtype)

    # Only rank 0 profiles and exports JSON
    if rank == 0:
        os.makedirs("./profiler_traces", exist_ok=True)
        trace_file = f"./profiler_traces/allreduce_{impl_name}_trace.json"

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
            # Profile dist.all_reduce
            for _ in range(num_iters):
                dist.all_reduce(tensor_dist, op=dist.ReduceOp.SUM)
            torch.xpu.synchronize()
            dist.barrier()

            # Profile specified implementation
            for _ in range(num_iters):
                impl_func(tensor_impl, op="sum")
            torch.xpu.synchronize()
            dist.barrier()

        # Export to JSON (Chrome trace format)
        prof.export_chrome_trace(trace_file)

        print(f"\nProfiler trace saved to: {trace_file}")
        print("Open with: chrome://tracing or https://ui.perfetto.dev/")
    else:
        # Other ranks run without profiling
        for _ in range(num_iters):
            dist.all_reduce(tensor_dist, op=dist.ReduceOp.SUM)
        torch.xpu.synchronize()
        dist.barrier()

        for _ in range(num_iters):
            impl_func(tensor_impl, op="sum")
        torch.xpu.synchronize()
        dist.barrier()

    # Barrier to sync all ranks
    dist.barrier()


def benchmark_allreduce(tensor_size, impl_func, num_warmup=10, num_iters=100, dtype=torch.bfloat16):
    """Benchmark the specified implementation against dist.all_reduce."""
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"

    tensor_size = (tensor_size // world_size) * world_size
    results = {}

    # Benchmark torch.distributed.all_reduce
    tensor_dist = torch.randn(tensor_size, device=device, dtype=dtype)
    for _ in range(num_warmup):
        t = tensor_dist.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.xpu.synchronize()

    dist.barrier()
    start = time.perf_counter()
    for _ in range(num_iters):
        dist.all_reduce(tensor_dist, op=dist.ReduceOp.SUM)
    torch.xpu.synchronize()
    end = time.perf_counter()
    results["dist.all_reduce"] = (end - start) / num_iters * 1000

    # Benchmark specified implementation
    tensor_impl = torch.randn(tensor_size, device=device, dtype=dtype)
    for _ in range(num_warmup):
        t = tensor_impl.clone()
        impl_func(t, op="sum")
    torch.xpu.synchronize()

    dist.barrier()
    start = time.perf_counter()
    for _ in range(num_iters):
        impl_func(tensor_impl, op="sum")
    torch.xpu.synchronize()
    end = time.perf_counter()
    results["impl"] = (end - start) / num_iters * 1000

    # Verify correctness - use same seed across ranks for same initial data
    torch.manual_seed(42 + rank)
    tensor_ref = torch.randn(tensor_size, device=device, dtype=torch.float32)
    tensor_test = tensor_ref.clone()

    dist.all_reduce(tensor_ref, op=dist.ReduceOp.SUM)
    impl_func(tensor_test, op="sum")

    torch.xpu.synchronize()

    is_correct = torch.allclose(tensor_ref, tensor_test, rtol=1e-3, atol=1e-3)

    return results, is_correct


def run_benchmark(impl_name: str, dtype=torch.bfloat16):
    """Run performance benchmark."""
    rank, world_size = init_distributed()
    impl_func = get_impl_func(impl_name)

    # Test tensor sizes: 8MB, 16MB, 32MB (BF16 = 2 bytes per element)
    sizes = [
        4194304,   # 8MB = 4M elements * 2 bytes
        8388608,   # 16MB = 8M elements * 2 bytes
        16777216,  # 32MB = 16M elements * 2 bytes
        33554432,  # 64MB = 32M elements * 2 bytes
        67108864,  # 128MB = 64M elements * 2 bytes
        134217728, # 256MB = 128M elements * 2 bytes
    ]

    # Get element size in bytes for the dtype
    element_size = torch.tensor([], dtype=dtype).element_size()

    if rank == 0:
        print("=" * 80)
        print(f"AllReduce Benchmark: {impl_name} (world_size={world_size}, dtype={dtype})")
        print("=" * 80)
        print(f"{'Size (MB)':>12} | {'dist.all_reduce':>15} | {impl_name:>15} | {'Speedup':>10} | {'Correct':>8}")
        print("-" * 80)

    for size in sizes:
        results, is_correct = benchmark_allreduce(size, impl_func, dtype=dtype)

        if rank == 0:
            dist_time = results["dist.all_reduce"]
            impl_time = results["impl"]
            speedup = dist_time / impl_time if impl_time > 0 else 0
            size_mb = size * element_size / (1024 * 1024)
            print(f"{size_mb:>12.1f} | {dist_time:>12.3f} ms | {impl_time:>12.3f} ms | "
                  f"{speedup:>9.2f}x | {'✓' if is_correct else '✗':>8}")

    if rank == 0:
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test allreduce implementations")
    parser.add_argument("--profile", action="store_true", help="Run with torch profiler")
    parser.add_argument("--accuracy", action="store_true", help="Run detailed accuracy check")
    parser.add_argument("--impl", type=str, default="symm_mem",
                        choices=list(IMPL_MAP.keys()),
                        help=f"Implementation to test: {list(IMPL_MAP.keys())}")
    args = parser.parse_args()

    if args.profile:
        run_profiler(args.impl)
    elif args.accuracy:
        run_accuracy_check(args.impl)
    else:
        run_benchmark(args.impl)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

