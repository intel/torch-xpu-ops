"""
Performance comparison: allreduce_with_symm_mem vs torch.distributed.all_reduce

Usage:
    mpirun -n 2 python test_allreduce.py
    mpirun -n 2 python test_allreduce.py --profile
    mpirun -n 2 python test_allreduce.py --accuracy
"""

import argparse
import time
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity
import os

from allreduce_impl import allreduce_with_symm_mem


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


def check_accuracy(tensor_size, rank, device, dtype=torch.bfloat16):
    """Detailed accuracy check for allreduce_with_symm_mem."""
    world_size = dist.get_world_size()
    tensor_size = (tensor_size // world_size) * world_size

    # Use same seed across ranks for reproducibility, but different data
    torch.manual_seed(42 + rank)
    tensor_ref = torch.randn(tensor_size, device=device, dtype=dtype)
    tensor_test = tensor_ref.clone()

    # Reference: torch.distributed.all_reduce
    dist.all_reduce(tensor_ref, op=dist.ReduceOp.SUM)

    # Test: our implementation
    allreduce_with_symm_mem(tensor_test, op="sum")

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


def run_accuracy_check():
    """Run detailed accuracy check."""
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"

    sizes = [1048576, 4194304, 8388608]

    if rank == 0:
        print("=" * 80)
        print(f"Accuracy Check (world_size={world_size})")
        print("=" * 80)
        print(f"{'Size':>12} | {'Pass':>6} | {'MaxAbsDiff':>12} | {'MeanAbsDiff':>12} | {'MaxRelDiff':>12}")
        print("-" * 80)

    for size in sizes:
        metrics = check_accuracy(size, rank, device)

        if rank == 0:
            status = "✓" if metrics["is_close"] else "✗"
            print(f"{size:>12} | {status:>6} | {metrics['max_abs_diff']:>12.2e} | "
                  f"{metrics['mean_abs_diff']:>12.2e} | {metrics['max_rel_diff']:>12.2e}")

    if rank == 0:
        print("=" * 80)


def run_profiler(dtype=torch.bfloat16):
    """Run with torch profiler. Only rank 0 generates JSON trace file."""
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"

    tensor_size = 8388608  # 16MB with BF16 (8M elements * 2 bytes)
    tensor_size = (tensor_size // world_size) * world_size
    num_iters = 10

    if rank == 0:
        print(f"Profiling both implementations (size={tensor_size}, dtype={dtype}, world_size={world_size})", flush=True)

    # Warmup both implementations
    for _ in range(5):
        t = torch.randn(tensor_size, device=device, dtype=dtype)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = torch.randn(tensor_size, device=device, dtype=dtype)
        allreduce_with_symm_mem(t, op="sum")
    torch.xpu.synchronize()

    # Pre-allocate tensors outside profiling loop
    tensor_dist_list = []
    tensor_symm_list = []
    for i in range(num_iters):
        tensor_dist_list.append(torch.randn(tensor_size, device=device, dtype=dtype))
        tensor_symm_list.append(torch.randn(tensor_size, device=device, dtype=dtype))

    torch.xpu.synchronize()
    # Only rank 0 profiles and exports JSON
    if rank == 0:
        import os
        os.makedirs("./profiler_traces", exist_ok=True)
        trace_file = "./profiler_traces/allreduce_trace_" + str(rank) + ".json"

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
            # Profile dist.all_reduce
            for m in range(num_iters):
                dist.all_reduce(tensor_dist_list[m], op=dist.ReduceOp.SUM)
            torch.xpu.synchronize()
            dist.barrier()

            # Profile allreduce_with_symm_mem
            for n in range(num_iters):
                allreduce_with_symm_mem(tensor_symm_list[n], op="sum")
            torch.xpu.synchronize()
            dist.barrier()

        # Export to JSON (Chrome trace format)
        prof.export_chrome_trace(trace_file)

        print(f"\nProfiler trace saved to: {trace_file}", flush=True)
        print("Open with: chrome://tracing or https://ui.perfetto.dev/", flush=True)
    # Barrier to sync all ranks
    dist.barrier()


def benchmark_allreduce(tensor_size, num_warmup=10, num_iters=100, dtype=torch.bfloat16):
    """Benchmark both implementations."""
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
        t = tensor_dist.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.xpu.synchronize()
    end = time.perf_counter()
    results["dist.all_reduce"] = (end - start) / num_iters * 1000

    # Benchmark allreduce_with_symm_mem
    tensor_symm = torch.randn(tensor_size, device=device, dtype=dtype)
    for _ in range(num_warmup):
        t = tensor_symm.clone()
        allreduce_with_symm_mem(t, op="sum")
    torch.xpu.synchronize()

    dist.barrier()
    start = time.perf_counter()
    for _ in range(num_iters):
        t = tensor_symm.clone()
        allreduce_with_symm_mem(t, op="sum")
    torch.xpu.synchronize()
    end = time.perf_counter()
    results["symm_mem"] = (end - start) / num_iters * 1000

    # Verify correctness - use same seed across ranks for same initial data
    torch.manual_seed(42)
    tensor_ref = torch.randn(tensor_size, device=device, dtype=dtype)
    tensor_test = tensor_ref.clone()

    dist.all_reduce(tensor_ref, op=dist.ReduceOp.SUM)
    allreduce_with_symm_mem(tensor_test, op="sum")

    is_correct = torch.allclose(tensor_ref, tensor_test, rtol=1e-3, atol=1e-3)  # BF16 tolerance

    return results, is_correct


def run_benchmark(dtype=torch.bfloat16):
    """Run performance benchmark."""
    rank, world_size = init_distributed()

    # Test tensor sizes: 8MB, 16MB, 32MB (BF16 = 2 bytes per element)
    sizes = [
        4194304,   # 8MB = 4M elements * 2 bytes
        8388608,   # 16MB = 8M elements * 2 bytes
        16777216,  # 32MB = 16M elements * 2 bytes
    ]

    if rank == 0:
        print("=" * 70)
        print(f"AllReduce Performance Comparison (world_size={world_size}, dtype={dtype})")
        print("=" * 70)
        print(f"{'Size':>12} | {'dist.all_reduce':>15} | {'symm_mem':>15} | {'Speedup':>10} | {'Correct':>8}")
        print("-" * 70)

    for size in sizes:
        results, is_correct = benchmark_allreduce(size, dtype=dtype)

        if rank == 0:
            dist_time = results["dist.all_reduce"]
            symm_time = results["symm_mem"]
            speedup = dist_time / symm_time if symm_time > 0 else 0
            print(f"{size:>12} | {dist_time:>12.3f} ms | {symm_time:>12.3f} ms | "
                  f"{speedup:>9.2f}x | {'✓' if is_correct else '✗':>8}")

    if rank == 0:
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test allreduce implementations")
    parser.add_argument("--profile", action="store_true", help="Run with torch profiler")
    parser.add_argument("--accuracy", action="store_true", help="Run detailed accuracy check")
    args = parser.parse_args()

    if args.profile:
        run_profiler()
    elif args.accuracy:
        run_accuracy_check()
    else:
        run_benchmark()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

