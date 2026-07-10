"""
Minimal repro for the ISHMEM mlx5dv / direct-doorbell failure path.

Usage:
    ZE_AFFINITY_MASK=4,5,6,7 mpirun -np 4 --prepend-rank \
        python test_allgather_permute_ishmem_mlx5dv_repro.py
"""

import os
import sys
import time

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

from allgather_local_permute_fusion import (
    allgather_permute_ishmem,
    allgather_permute_ishmem_finalize,
    compute_scatter_idx,
)


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

    num_tokens_per_rank = int(os.environ.get("TOKENS_PER_RANK", "4"))
    hidden_size = int(os.environ.get("HIDDEN_SIZE", "16"))
    topk = int(os.environ.get("TOPK", "2"))
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

    # ---------------- Performance measurement ----------------
    warmup = int(os.environ.get("PERF_WARMUP", "20"))
    iters = int(os.environ.get("PERF_ITERS", "100"))

    print("[perf] starting performance measurement", flush=True)
    for _ in range(warmup):
        allgather_permute_ishmem(hidden_shard, scatter_idx, output, group=dist.group.WORLD)
    torch.xpu.synchronize()
    dist.barrier()

    per_iter_ms = []
    for _ in range(iters):
        torch.xpu.synchronize()
        dist.barrier()
        start = time.perf_counter()
        allgather_permute_ishmem(hidden_shard, scatter_idx, output, group=dist.group.WORLD)
        torch.xpu.synchronize()
        end = time.perf_counter()
        per_iter_ms.append((end - start) * 1e3)
    print(f"[perf] completed {iters} iterations", flush=True)
    per_iter_ms.sort()
    avg_ms = sum(per_iter_ms) / len(per_iter_ms)
    p50_ms = per_iter_ms[len(per_iter_ms) // 2]
    p99_ms = per_iter_ms[min(len(per_iter_ms) - 1, int(len(per_iter_ms) * 0.99))]
    min_ms = per_iter_ms[0]
    max_ms = per_iter_ms[-1]

    # Per-PE NIC traffic per call: this PE pushes its shard to (world_size-1) peers.
    shard_bytes = hidden_shard.numel() * hidden_shard.element_size()
    nic_bytes_per_iter = shard_bytes * (world_size - 1)
    nic_gbps = (nic_bytes_per_iter / (avg_ms * 1e-3)) / 1e9 if avg_ms > 0 else 0.0

    stats = torch.tensor([avg_ms, min_ms, max_ms, p50_ms, p99_ms], device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if rank == 0:
        print(
            "[perf] allgather_permute_ishmem  "
            f"world_size={world_size} tokens/rank={num_tokens_per_rank} "
            f"hidden={hidden_size} topk={topk} dtype={hidden_shard.dtype}",
            flush=True,
        )
        print(
            f"[perf] iters={iters} warmup={warmup}  "
            f"shard={shard_bytes/1024:.1f}KiB  nic_push/iter={nic_bytes_per_iter/1024:.1f}KiB",
            flush=True,
        )
        print(
            "[perf] latency ms (max over ranks): "
            f"avg={stats[0].item():.4f} min={stats[1].item():.4f} "
            f"max={stats[2].item():.4f} p50={stats[3].item():.4f} p99={stats[4].item():.4f}",
            flush=True,
        )
        print(
            f"[perf] per-PE NIC push BW ~= {nic_gbps:.2f} GB/s (avg-latency based)",
            flush=True,
        )
    # ---------------------------------------------------------

    # Cleanly stop the ISHMEM runtime/proxy thread before MPI teardown to avoid
    # a crash-on-exit. This is collective across all ranks.
    dist.barrier()
    allgather_permute_ishmem_finalize(device)
    dist.barrier()


if __name__ == "__main__":
    main()
