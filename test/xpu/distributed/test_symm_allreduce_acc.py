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

from allreduce_impl import allreduce_with_symm_mem,allreduce_cross_switch


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


def check_accuracy(tensor_size, rank, device, dtype=torch.float32):
    """Detailed accuracy check for allreduce_with_symm_mem."""
    world_size = dist.get_world_size()
    tensor_size = (tensor_size // world_size) * world_size

    # Use same seed across ranks for reproducibility, but different data
    torch.manual_seed(42 + rank)
    tensor_ref = torch.randn(tensor_size, device=device, dtype=dtype)
    # tensor_ref = torch.ones(tensor_size, device=device) * rank
    tensor_test = tensor_ref.clone()

    # Reference: torch.distributed.all_reduce
    dist.all_reduce(tensor_ref, op=dist.ReduceOp.SUM)

    # Test: our implementation
    allreduce_cross_switch(tensor_test, op="sum")

    torch.xpu.synchronize()

    # print(f"dist allreduce = {tensor_ref} symm_mem allreduce = {tensor_test} \n", flush=True)

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

    sizes = [
        4194304,   # 8MB = 4M elements * 2 bytes
        8388608,   # 16MB = 8M elements * 2 bytes
        16777216,  # 32MB = 16M elements * 2 bytes
        # 33554432,  # 64MB = 32M elements * 2 bytes
        # 67108864,  # 128MB = 64M elements * 2 bytes
        # 134217728, # 256MB = 128M elements * 2 bytes
    ]

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


def main():
    run_accuracy_check()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

