"""
Accuracy + performance check for deepep_owner_combine (distributed style)

Usage:
    mpirun -n 2 python test_deepep_combine_dist.py
"""

import os

import torch
import torch.distributed as dist

from allgather_local_permute_fusion import compute_scatter_idx
from deepep_dispatch import (
    _HAS_EP_COMBINE_KERNEL,
    _HAS_EP_COMBINE_LOCAL,
    deepep_owner_combine,
    get_expert_owner,
)

TOKENS_PER_RANK = 4096
HIDDEN_SIZE = 5120
TOPK = 8
NUM_EXPERTS = 128
LOOP = 40
WARMUP = 20


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29520"
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def make_inputs(rank, world_size, device):
    num_tokens_per_rank = TOKENS_PER_RANK
    hidden_size = HIDDEN_SIZE
    topk = TOPK
    num_tokens = num_tokens_per_rank * world_size

    torch.manual_seed(42)
    topk_idx = torch.randint(
        0,
        NUM_EXPERTS,
        (num_tokens, topk),
        device=device,
        dtype=torch.int32,
    )
    scatter_idx, _ = compute_scatter_idx(topk_idx, num_experts=NUM_EXPERTS)

    torch.manual_seed(777)
    topk_weights = torch.rand(num_tokens, topk, device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    torch.manual_seed(123)
    expert_output_full = torch.randn(
        num_tokens * topk,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )

    # Simulate EP-owner local expert outputs: each rank keeps only rows it owns.
    expert_output_local = torch.zeros_like(expert_output_full)
    for i in range(num_tokens):
        for k in range(topk):
            expert = int(topk_idx[i, k].item())
            owner = get_expert_owner(expert, NUM_EXPERTS, world_size)
            if owner == rank:
                row = int(scatter_idx[i, k].item())
                expert_output_local[row].copy_(expert_output_full[row])

    return expert_output_full, expert_output_local, topk_idx, scatter_idx, topk_weights


def build_reference_shard(expert_output_full, scatter_idx, topk_weights, rank, world_size):
    num_tokens, topk = scatter_idx.shape
    hidden = expert_output_full.shape[1]
    num_tokens_per_rank = num_tokens // world_size

    # Accumulate in float32 for precision, then convert back
    full_result = torch.zeros(num_tokens, hidden, device=expert_output_full.device, dtype=torch.float32)
    for i in range(num_tokens):
        for k in range(topk):
            src_row = int(scatter_idx[i, k].item())
            full_result[i] += topk_weights[i, k] * expert_output_full[src_row].float()

    my_start = rank * num_tokens_per_rank
    return full_result[my_start: my_start + num_tokens_per_rank].to(expert_output_full.dtype)


def check_deepep_owner_combine():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    num_tokens_per_rank = TOKENS_PER_RANK
    hidden_size = HIDDEN_SIZE

    expert_output_full, expert_output_local, topk_idx, scatter_idx, topk_weights = make_inputs(
        rank, world_size, device
    )

    rank_output_ptrs = None  # Pipeline approach doesn't need precomputed pointers

    backend_stream = torch.xpu.Stream()

    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    # Warm up
    for _ in range(WARMUP):
        out = torch.zeros(num_tokens_per_rank, hidden_size, device=device, dtype=expert_output_local.dtype)
        deepep_owner_combine(
            expert_output=expert_output_local,
            topk_idx=topk_idx,
            scatter_idx=scatter_idx,
            topk_weights=topk_weights,
            output=out,
            num_experts=NUM_EXPERTS,
            group=group,
            backend_stream=backend_stream,
            rank_output_ptrs=rank_output_ptrs,
        )
    torch.xpu.synchronize()
    dist.barrier()

    # Timed
    for i in range(LOOP):
        if i >= WARMUP:
            begin_events[i].record()
        out = torch.zeros(num_tokens_per_rank, hidden_size, device=device, dtype=expert_output_local.dtype)
        deepep_owner_combine(
            expert_output=expert_output_local,
            topk_idx=topk_idx,
            scatter_idx=scatter_idx,
            topk_weights=topk_weights,
            output=out,
            num_experts=NUM_EXPERTS,
            group=group,
            backend_stream=backend_stream,
            rank_output_ptrs=rank_output_ptrs,
        )
        if i >= WARMUP:
            end_events[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    print(f"[DeePEP combine time in rank {rank}] {latencies} ms")

    # Accuracy check
    check_out = torch.zeros(num_tokens_per_rank, hidden_size, device=device, dtype=expert_output_local.dtype)
    deepep_owner_combine(
        expert_output=expert_output_local,
        topk_idx=topk_idx,
        scatter_idx=scatter_idx,
        topk_weights=topk_weights,
        output=check_out,
        num_experts=NUM_EXPERTS,
        group=group,
        backend_stream=backend_stream,
        rank_output_ptrs=rank_output_ptrs,
    )

    ref_shard = build_reference_shard(
        expert_output_full=expert_output_full,
        scatter_idx=scatter_idx,
        topk_weights=topk_weights,
        rank=rank,
        world_size=world_size,
    )

    assert torch.allclose(check_out, ref_shard, atol=2e-2, rtol=2e-2), (
        f"deepep_owner_combine mismatch in rank {rank}, "
        f"max_diff={(check_out - ref_shard).abs().max().item():.6f}"
    )

    if rank == 0:
        avg = sum(latencies) / len(latencies)
        print(f"[Summary] deepep_owner_combine avg={avg:.3f} ms")
        if _HAS_EP_COMBINE_LOCAL:
            print("[Summary] kernel=pipeline (ep_combine_local + push)")
        elif _HAS_EP_COMBINE_KERNEL:
            print("[Summary] kernel=ep_combine (single fused kernel)")
        else:
            print("[Summary] kernel=no (Python fallback)")

    dist.destroy_process_group()


if __name__ == "__main__":
    check_deepep_owner_combine()
