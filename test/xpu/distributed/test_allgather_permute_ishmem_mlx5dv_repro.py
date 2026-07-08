"""
Minimal repro for the ISHMEM mlx5dv / direct-doorbell failure path.

Usage:
    ZE_AFFINITY_MASK=4,5,6,7 mpirun -np 4 --prepend-rank \
        python test_allgather_permute_ishmem_mlx5dv_repro.py
"""

import os

os.environ.setdefault("ISHMEM_IB_ENABLE_IBGDA", "1")
os.environ.setdefault("ISHMEM_IBGDA_DIRECT_DOORBELL", "1")
os.environ.setdefault("ISHMEM_ENABLE_GPU_IPC", "0")
os.environ.setdefault("ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP", "1")
os.environ.setdefault("ISHMEM_SYMMETRIC_SIZE", str(512 * 1024 * 1024))
os.environ.setdefault("ZE_ENABLE_PCI_ID_DEVICE_ORDER", "1")
os.environ.setdefault("ISHMEM_IBGDA_QPS_PER_PE", "1")
os.environ.setdefault("ISHMEM_IBGDA_DB_BATCH_SIZE", "0")
os.environ.setdefault("ISHMEM_IBGDA_BAR_BACKEND", "igub")
os.environ.setdefault("I_MPI_FABRICS", "shm")
os.environ.setdefault("ISHMEM_DEBUG", "0")

import torch
import torch.distributed as dist

from allgather_local_permute_fusion import allgather_permute_ishmem, compute_scatter_idx


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29531")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.xpu.device_count()
    torch.xpu.set_device(device_id)
    return rank, world_size, f"xpu:{device_id}"


def main():
    rank, world_size, device = init_distributed()
    if rank == 0:
        print(
            "[mlx5dv repro] forcing ISHMEM IBGDA direct-doorbell path",
            flush=True,
        )

    num_tokens_per_rank = 4
    hidden_size = 16
    topk = 2
    num_tokens = num_tokens_per_rank * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        num_tokens_per_rank,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(42)
    topk_idx_cpu = torch.randint(
        0,
        8,
        (num_tokens, topk),
        generator=cpu_generator,
        dtype=torch.int32,
    )
    scatter_idx_cpu, _ = compute_scatter_idx(topk_idx_cpu, num_experts=8)
    scatter_idx = scatter_idx_cpu.to(device).contiguous()

    output = torch.empty(
        (num_tokens * topk, hidden_size),
        device=device,
        dtype=hidden_shard.dtype,
    )

    if rank == 0:
        print("[mlx5dv repro] invoking allgather_permute_ishmem", flush=True)

    allgather_permute_ishmem(hidden_shard, scatter_idx, output, group=dist.group.WORLD)
    torch.xpu.synchronize()
    dist.barrier()

    if rank == 0:
        print("[mlx5dv repro] completed without hitting the expected failure", flush=True)


if __name__ == "__main__":
    main()
