"""
Accuracy + performance check for deepep_owner_dispatch (distributed style)

Usage:
    mpirun -n 2 python test_deepep_dispatch.py
"""

import os

import torch
import torch.distributed as dist

from deepep_dispatch import deepep_owner_dispatch, get_expert_owner


TOKENS_PER_RANK = 256
HIDDEN_SIZE = 512
TOPK = 4
NUM_EXPERTS = 8
LOOP = 10


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

    # Warm up
    for _ in range(LOOP):
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

    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    remap_hidden_states = torch.zeros(
        (num_tokens * topk, hidden_size),
        device=device,
        dtype=hidden_shard.dtype,
    )
    for i in range(LOOP):
        begin_events[i].record()
        deepep_owner_dispatch(
            hidden_shard,
            topk_idx,
            remap_hidden_states,
            num_experts=NUM_EXPERTS,
            group=group,
        )
        end_events[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]
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

    dist.destroy_process_group()


if __name__ == "__main__":
    check_deepep_owner_dispatch()
