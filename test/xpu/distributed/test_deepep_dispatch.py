"""
Accuracy + performance check for deepep_owner_dispatch (distributed style)

Usage:
    mpirun -n 2 python test_deepep_dispatch.py
"""

import os

import torch
import torch.distributed as dist

from deepep_dispatch import deepep_owner_dispatch, get_expert_owner, build_rank_buffers_ptr

TOKENS_PER_RANK = 2048
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
LOOP = 40
WARMUP = 20
ENABLE_PROJECTION = False
CROSS_GPU_BW_GBPS = 30.0
HBM_BW_GBPS = 1300.0


def bytes_to_mb(num_bytes: float) -> float:
    return num_bytes / (1024 * 1024)


def project_time_ms(bytes_count: float, bw_gbps: float) -> float:
    # GB/s is interpreted as 1e9 bytes/s for bandwidth projection.
    return bytes_count / (bw_gbps * 1e9) * 1e3


def build_dispatch_projection(
    topk_idx: torch.Tensor,
    num_tokens_per_rank: int,
    hidden_size: int,
    elem_size: int,
    num_experts: int,
    world_size: int,
):
    num_tokens, topk = topk_idx.shape

    owner_table = torch.tensor(
        [get_expert_owner(e, num_experts, world_size) for e in range(num_experts)],
        device=topk_idx.device,
        dtype=torch.int64,
    )
    owners = owner_table[topk_idx.reshape(-1)]

    token_ids = torch.arange(num_tokens, device=topk_idx.device, dtype=torch.int64)
    src_ranks = (token_ids // num_tokens_per_rank).repeat_interleave(topk)

    pair_ids = src_ranks * world_size + owners
    pair_counts = torch.bincount(pair_ids, minlength=world_size * world_size).reshape(world_size, world_size)

    diag = torch.diagonal(pair_counts)
    send_counts = pair_counts.sum(dim=1) - diag
    recv_counts = pair_counts.sum(dim=0) - diag
    local_assign_counts = pair_counts.sum(dim=0)

    bytes_per_assignment = hidden_size * elem_size

    send_bytes = send_counts.to(torch.float64) * bytes_per_assignment
    recv_bytes = recv_counts.to(torch.float64) * bytes_per_assignment
    comm_bytes = torch.maximum(send_bytes, recv_bytes)

    # One read + one write for local remap data movement.
    local_permute_bytes = local_assign_counts.to(torch.float64) * bytes_per_assignment * 2

    comm_ms = comm_bytes / (CROSS_GPU_BW_GBPS * 1e9) * 1e3
    local_ms = local_permute_bytes / (HBM_BW_GBPS * 1e9) * 1e3

    return {
        "send_bytes": send_bytes,
        "recv_bytes": recv_bytes,
        "comm_ms": comm_ms,
        "local_permute_bytes": local_permute_bytes,
        "local_ms": local_ms,
        "fused_ms": comm_ms + local_ms,
    }


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29519"
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def build_reference(
    all_hidden: torch.Tensor,
    topk_idx: torch.Tensor,
    num_experts: int,
    rank: int,
    world_size: int,
):
    num_tokens_per_rank, hidden_size = all_hidden.shape[1], all_hidden.shape[2]
    num_tokens, topk = topk_idx.shape
    ref = torch.zeros(
        (num_tokens * topk, hidden_size),
        device=all_hidden.device,
        dtype=all_hidden.dtype,
    )

    for src_rank in range(world_size):
        for i in range(num_tokens_per_rank):
            global_token_idx = src_rank * num_tokens_per_rank + i
            for k in range(topk):
                expert = int(topk_idx[global_token_idx, k].item())
                owner = get_expert_owner(expert, num_experts, world_size)
                if owner == rank:
                    dst = global_token_idx * topk + k
                    ref[dst].copy_(all_hidden[src_rank, i])
    return ref


def check_deepep_owner_dispatch():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    num_tokens_per_rank = TOKENS_PER_RANK
    hidden_size = HIDDEN_SIZE
    topk = TOPK
    num_tokens = num_tokens_per_rank * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        num_tokens_per_rank,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )

    torch.manual_seed(42)
    topk_idx = torch.randint(
        0,
        NUM_EXPERTS,
        (num_tokens, topk),
        device=device,
        dtype=torch.int64,
    )

    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    # Warm up
    for _ in range(WARMUP):
        remap_hidden_states = torch.zeros(
            (num_tokens * topk, hidden_size),
            device=device,
            dtype=hidden_shard.dtype,
        )
        deepep_owner_dispatch(
            hidden_shard,
            topk_idx,
            remap_hidden_states,
            num_experts=NUM_EXPERTS,
            group=group,
        )
    torch.xpu.synchronize()
    dist.barrier()

    # Precompute rank_buffers_ptr for timed path (pointers are stable after warmup)
    rank_buffers = build_rank_buffers_ptr(hidden_shard, NUM_EXPERTS, group=group)

    remap_hidden_states = torch.zeros(
        (num_tokens * topk, hidden_size),
        device=device,
        dtype=hidden_shard.dtype,
    )
    # Timed path
    for i in range(LOOP):
        if i >= WARMUP:
            begin_events[i].record()
        deepep_owner_dispatch(
            hidden_shard,
            topk_idx,
            remap_hidden_states,
            num_experts=NUM_EXPERTS,
            group=group,
            rank_buffers_ptr=rank_buffers,
        )
        if i >= WARMUP:
            end_events[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    print(f"[DeePEP dispatch time in rank {rank}] {latencies} ms")

    gathered_hidden = [torch.empty_like(hidden_shard) for _ in range(world_size)]
    dist.all_gather(gathered_hidden, hidden_shard, group=group)
    all_hidden = torch.stack(gathered_hidden, dim=0)
    ref = build_reference(all_hidden, topk_idx, NUM_EXPERTS, rank, world_size)

    assert torch.equal(
        remap_hidden_states,
        ref,
    ), f"deepep_owner_dispatch mismatch in rank {rank}"

    if rank == 0:
        avg = sum(latencies) / len(latencies)
        print(f"[Summary] deepep_owner_dispatch avg={avg:.3f} ms")

        if ENABLE_PROJECTION:
            projection = build_dispatch_projection(
                topk_idx=topk_idx,
                num_tokens_per_rank=num_tokens_per_rank,
                hidden_size=hidden_size,
                elem_size=hidden_shard.element_size(),
                num_experts=NUM_EXPERTS,
                world_size=world_size,
            )

            send_bytes = projection["send_bytes"].to("cpu").tolist()
            recv_bytes = projection["recv_bytes"].to("cpu").tolist()
            comm_ms = projection["comm_ms"].to("cpu").tolist()
            local_permute_bytes = projection["local_permute_bytes"].to("cpu").tolist()
            local_ms = projection["local_ms"].to("cpu").tolist()
            fused_ms = projection["fused_ms"].to("cpu").tolist()

            for r in range(world_size):
                print(
                    f"[Projection][rank {r}] "
                    f"send={bytes_to_mb(send_bytes[r]):.2f} MB, "
                    f"recv={bytes_to_mb(recv_bytes[r]):.2f} MB, "
                    f"comm@{CROSS_GPU_BW_GBPS:.1f}GB/s={comm_ms[r]:.3f} ms"
                )
                print(
                    f"[Projection][rank {r}] "
                    f"local_permute={bytes_to_mb(local_permute_bytes[r]):.2f} MB, "
                    f"hbm@{HBM_BW_GBPS:.1f}GB/s={local_ms[r]:.3f} ms, "
                    f"fused_lower_bound={fused_ms[r]:.3f} ms"
                )

            worst_rank = max(range(world_size), key=lambda r: fused_ms[r])
            print(
                f"[Projection][Summary] worst_rank={worst_rank}, "
                f"fused_lower_bound={fused_ms[worst_rank]:.3f} ms"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    check_deepep_owner_dispatch()
