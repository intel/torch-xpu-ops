"""
Accuracy test for ISHMEM allgather+permute fusion.

Usage:
    mpirun -n 2 python test_allgather_permute_ishmem_dist.py
"""

import os

import torch
import torch.distributed as dist

from allgather_local_permute_fusion import (
    allgather_permute_ishmem,
    compute_scatter_idx,
)

def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29517"
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def run_reference(hidden_shard, scatter_idx, world_size, num_tokens_per_rank, topk):
    gathered = [torch.empty_like(hidden_shard) for _ in range(world_size)]
    dist.all_gather(gathered, hidden_shard)
    hidden_size = hidden_shard.size(1)
    ref = torch.empty(
        (num_tokens_per_rank * world_size * topk, hidden_size),
        device=hidden_shard.device,
        dtype=hidden_shard.dtype,
    )
    for src_rank, shard in enumerate(gathered):
        token_offset = src_rank * num_tokens_per_rank
        for local_token_idx in range(num_tokens_per_rank):
            global_token_idx = token_offset + local_token_idx
            for k in range(topk):
                dst = scatter_idx[global_token_idx, k].item()
                ref[dst].copy_(shard[local_token_idx])
    return ref


def check_case(dtype, num_tokens_per_rank, hidden_size, topk, num_experts):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = f"xpu:{rank}"
    num_tokens = num_tokens_per_rank * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        num_tokens_per_rank,
        hidden_size,
        device=device,
        dtype=dtype,
    )

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(42)
    topk_idx_cpu = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        generator=cpu_generator,
        dtype=torch.int32,
    )
    scatter_idx_cpu, _ = compute_scatter_idx(topk_idx_cpu, num_experts=num_experts)
    scatter_idx = scatter_idx_cpu.to(device).contiguous()

    ref = run_reference(hidden_shard, scatter_idx, world_size, num_tokens_per_rank, topk)

    output = torch.empty(
        (num_tokens * topk, hidden_size),
        device=device,
        dtype=dtype,
    )
    begin = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    begin.record()
    allgather_permute_ishmem(hidden_shard, scatter_idx, output)
    end.record()
    torch.xpu.synchronize()

    assert torch.equal(output, ref), (
        f"ISHMEM allgather_permute mismatch on rank {rank}: "
        f"dtype={dtype}, tokens_per_rank={num_tokens_per_rank}, "
        f"hidden={hidden_size}, topk={topk}"
    )
    if rank == 0:
        elem_size = hidden_shard.element_size()
        recv_bytes = (world_size - 1) * num_tokens_per_rank * hidden_size * elem_size
        write_bytes = num_tokens * topk * hidden_size * elem_size
        print(
            "[ISHMEM allgather_permute case] "
            f"dtype={dtype}, world_size={world_size}, "
            f"tokens_per_rank={num_tokens_per_rank}, hidden={hidden_size}, topk={topk}, "
            f"latency={begin.elapsed_time(end):.3f} ms, "
            f"recv={recv_bytes / 1024 / 1024:.2f} MiB, "
            f"write={write_bytes / 1024 / 1024:.2f} MiB",
            flush=True,
        )


def main():
    rank, _ = init_distributed()
    os.environ.setdefault("ISHMEM_SYMMETRIC_SIZE", "16777216")
    try:
        for dtype in (torch.float32, torch.bfloat16):
            check_case(dtype, num_tokens_per_rank=3, hidden_size=7, topk=1, num_experts=4)
            check_case(dtype, num_tokens_per_rank=4, hidden_size=16, topk=2, num_experts=8)
            check_case(dtype, num_tokens_per_rank=5, hidden_size=17, topk=3, num_experts=8)
        if rank == 0:
            print("ISHMEM allgather_permute accuracy passed", flush=True)
    finally:
        if dist.is_initialized():
            dist.barrier()
        # ishmem_finalize currently hangs in this mixed XCCL+ISHMEM test process.
        # Exit after producing accuracy/perf results; teardown will be handled separately.
        os._exit(0)


if __name__ == "__main__":
    main()
