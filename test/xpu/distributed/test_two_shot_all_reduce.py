"""
Accuracy test for two_shot_all_reduce_ and two_shot_all_reduce_out operations.

Usage:
    mpirun -n 2 python test_two_shot_all_reduce.py
    mpirun -n 2 python test_two_shot_all_reduce.py --impl inplace
    mpirun -n 2 python test_two_shot_all_reduce.py --impl out

The test uses dist.all_reduce as the reference to verify accuracy.
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import _SymmetricMemory


def init_distributed():
    """Initialize distributed environment."""
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29514'
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)

    return rank, world_size


def create_symm_mem_tensor(size, dtype, device, group_name):
    """Create a tensor allocated with symmetric memory."""
    strides = torch._prims_common.make_contiguous_strides_for(size)
    return _SymmetricMemory.empty_strided_p2p(
        size,
        strides,
        dtype,
        device,
        group_name,
    )


def check_accuracy_inplace(tensor_size, rank, device, dtype=torch.float32):
    """Detailed accuracy check for two_shot_all_reduce_ (in-place)."""
    group = dist.group.WORLD

    # Use same seed across ranks for reproducibility, but different data
    torch.manual_seed(42 + rank)
    local_data = torch.randn(tensor_size, device=device, dtype=dtype)

    # Reference: torch.distributed.all_reduce
    tensor_ref = local_data.clone()
    dist.all_reduce(tensor_ref, op=dist.ReduceOp.SUM)

    # Test: two_shot_all_reduce_ (in-place)
    symm_tensor = create_symm_mem_tensor((tensor_size,), dtype, device, group.group_name)
    symm_tensor.copy_(local_data)
    tensor_test = torch.ops.symm_mem.two_shot_all_reduce_(
        symm_tensor, "sum", group.group_name
    )

    torch.xpu.synchronize()

    # Compute metrics
    abs_diff = torch.abs(tensor_ref - tensor_test)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    rel_diff = abs_diff / (torch.abs(tensor_ref) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    is_close = torch.allclose(tensor_ref, tensor_test, rtol=1e-3, atol=1e-3)

    return {
        "is_close": is_close,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
    }


def check_accuracy_out(tensor_size, rank, device, dtype=torch.float32):
    """Detailed accuracy check for two_shot_all_reduce_out."""
    group = dist.group.WORLD

    # Use same seed across ranks for reproducibility, but different data
    torch.manual_seed(42 + rank)
    local_data = torch.randn(tensor_size, device=device, dtype=dtype)

    # Reference: torch.distributed.all_reduce
    tensor_ref = local_data.clone()
    dist.all_reduce(tensor_ref, op=dist.ReduceOp.SUM)

    # Test: two_shot_all_reduce_out
    symm_tensor = create_symm_mem_tensor((tensor_size,), dtype, device, group.group_name)
    symm_tensor.copy_(local_data)
    output = torch.empty(tensor_size, device=device, dtype=dtype)
    tensor_test = torch.ops.symm_mem.two_shot_all_reduce_out(
        symm_tensor, "sum", group.group_name, output
    )

    torch.xpu.synchronize()

    # Compute metrics
    abs_diff = torch.abs(tensor_ref - tensor_test)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    rel_diff = abs_diff / (torch.abs(tensor_ref) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    is_close = torch.allclose(tensor_ref, tensor_test, rtol=1e-3, atol=1e-3)

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
    device = torch.device("xpu", rank)

    # Select check function based on implementation
    if impl_name == "inplace":
        check_func = check_accuracy_inplace
        impl_display = "two_shot_all_reduce_"
    else:
        check_func = check_accuracy_out
        impl_display = "two_shot_all_reduce_out"

    sizes = [64, 256, 1024, 4096, 8192, 65536, 262144, 1048576]
    dtypes = [torch.float32, torch.bfloat16]

    for dtype in dtypes:
        if rank == 0:
            print("=" * 80)
            print(f"Accuracy Check: {impl_display} (world_size={world_size}, dtype={dtype})")
            print("=" * 80)
            print(f"{'Size':>12} | {'Pass':>6} | {'MaxAbsDiff':>12} | {'MeanAbsDiff':>12} | {'MaxRelDiff':>12}")
            print("-" * 80)

        for size in sizes:
            metrics = check_func(size, rank, device, dtype=dtype)

            if rank == 0:
                status = "✓" if metrics["is_close"] else "✗"
                print(f"{size:>12} | {status:>6} | {metrics['max_abs_diff']:>12.2e} | "
                      f"{metrics['mean_abs_diff']:>12.2e} | {metrics['max_rel_diff']:>12.2e}")

        if rank == 0:
            print("=" * 80)
            print()


def main():
    parser = argparse.ArgumentParser(description="Test two_shot_all_reduce implementations")
    parser.add_argument("--impl", type=str, default="inplace",
                        choices=["inplace", "out"],
                        help="Implementation to test: inplace or out (default: inplace)")
    args = parser.parse_args()

    run_accuracy_check(args.impl)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

