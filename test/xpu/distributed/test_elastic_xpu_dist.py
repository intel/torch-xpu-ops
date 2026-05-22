"""
Accuracy + smoke test for elastic_xpu.ElasticBuffer on XPU.

This validates that the XPU wrapper matches the existing deepep_dispatch
reference paths for dispatch and combine.

Usage:
    mpirun -n 2 python test_elastic_xpu_dist.py
"""

import os

import torch
import torch.distributed as dist

from allgather_local_permute_fusion import compute_scatter_idx
from deepep_dispatch import (
    build_combine_rank_output_ptrs,
    deepep_owner_combine,
    get_expert_owner,
)
from elastic_xpu import EPHandle, ElasticBuffer

TOKENS_PER_RANK = 2048
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
LOOP = 20
WARMUP = 10


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29522"
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def make_dispatch_inputs(rank, world_size, device):
    num_tokens_per_rank = TOKENS_PER_RANK
    hidden_size = HIDDEN_SIZE
    topk = TOPK
    num_tokens = num_tokens_per_rank * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(num_tokens_per_rank, hidden_size, device=device, dtype=torch.bfloat16)

    torch.manual_seed(42)
    topk_idx = torch.randint(0, NUM_EXPERTS, (num_tokens, topk), device=device, dtype=torch.int32)
    scatter_idx, _ = compute_scatter_idx(topk_idx, num_experts=NUM_EXPERTS)

    torch.manual_seed(777)
    topk_weights = torch.rand(num_tokens, topk, device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    return hidden_shard, topk_idx, scatter_idx, topk_weights


def make_combine_inputs(rank, world_size, device):
    num_tokens_per_rank = TOKENS_PER_RANK
    hidden_size = HIDDEN_SIZE
    topk = TOPK
    num_tokens = num_tokens_per_rank * world_size

    torch.manual_seed(42)
    topk_idx = torch.randint(0, NUM_EXPERTS, (num_tokens, topk), device=device, dtype=torch.int32)
    scatter_idx, _ = compute_scatter_idx(topk_idx, num_experts=NUM_EXPERTS)

    torch.manual_seed(777)
    topk_weights = torch.rand(num_tokens, topk, device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    torch.manual_seed(123)
    expert_output_full = torch.randn(num_tokens * topk, hidden_size, device=device, dtype=torch.bfloat16)

    expert_output_local = torch.zeros_like(expert_output_full)
    for i in range(num_tokens):
        for k in range(topk):
            expert = int(topk_idx[i, k].item())
            owner = get_expert_owner(expert, NUM_EXPERTS, world_size)
            if owner == rank:
                row = int(scatter_idx[i, k].item())
                expert_output_local[row].copy_(expert_output_full[row])

    return expert_output_full, expert_output_local, topk_idx, scatter_idx, topk_weights


def build_dispatch_reference(hidden_shard, topk_idx, scatter_idx, num_experts, group):
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    gathered_hidden = [torch.empty_like(hidden_shard) for _ in range(world_size)]
    dist.all_gather(gathered_hidden, hidden_shard, group=group)
    all_hidden = torch.stack(gathered_hidden, dim=0)

    num_tokens_per_rank = hidden_shard.shape[0]
    hidden_size = hidden_shard.shape[1]
    num_tokens, topk = topk_idx.shape
    ref = torch.zeros(num_tokens * topk, hidden_size, device=hidden_shard.device, dtype=hidden_shard.dtype)

    for src_rank in range(world_size):
        for i in range(num_tokens_per_rank):
            global_token_idx = src_rank * num_tokens_per_rank + i
            for k in range(topk):
                expert = int(topk_idx[global_token_idx, k].item())
                owner = get_expert_owner(expert, num_experts, world_size)
                if owner == rank:
                    dst = int(scatter_idx[global_token_idx, k].item())
                    ref[dst].copy_(all_hidden[src_rank, i])
    return ref


def build_combine_reference(expert_output_full, scatter_idx, topk_weights, rank, world_size):
    num_tokens, topk = scatter_idx.shape
    hidden = expert_output_full.shape[1]
    num_tokens_per_rank = num_tokens // world_size
    full_result = torch.zeros(num_tokens, hidden, device=expert_output_full.device, dtype=expert_output_full.dtype)
    for i in range(num_tokens):
        for k in range(topk):
            src_row = int(scatter_idx[i, k].item())
            full_result[i] += topk_weights[i, k] * expert_output_full[src_row]
    my_start = rank * num_tokens_per_rank
    return full_result[my_start : my_start + num_tokens_per_rank]


def check_elastic_xpu_dispatch_and_combine():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    hidden_shard, topk_idx, scatter_idx, topk_weights = make_dispatch_inputs(rank, world_size, device)
    expert_output_full, expert_output_local, topk_idx_c, scatter_idx_c, topk_weights_c = make_combine_inputs(
        rank, world_size, device
    )

    assert torch.equal(topk_idx, topk_idx_c)
    assert torch.equal(scatter_idx, scatter_idx_c)
    assert torch.allclose(topk_weights, topk_weights_c)

    buffer = ElasticBuffer(
        group=group,
        num_max_tokens_per_rank=TOKENS_PER_RANK,
        hidden=HIDDEN_SIZE,
        num_topk=TOPK,
    )

    # Dispatch should match the deepep reference path.
    dispatch_out = torch.zeros_like(torch.zeros(TOKENS_PER_RANK * world_size * TOPK, HIDDEN_SIZE, device=device, dtype=torch.bfloat16))
    ref_dispatch_out = torch.zeros_like(dispatch_out)
    handle = None

    for _ in range(WARMUP):
        dispatch_out.zero_()
        dispatch_out, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
            hidden_shard,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=NUM_EXPERTS,
            do_cpu_sync=True,
        )
    torch.xpu.synchronize()
    dist.barrier()

    for _ in range(LOOP):
        dispatch_out.zero_()
        dispatch_out, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(
            hidden_shard,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=NUM_EXPERTS,
            do_cpu_sync=True,
        )
    torch.xpu.synchronize()
    dist.barrier()

    ref_dispatch_out = build_dispatch_reference(hidden_shard, topk_idx, scatter_idx, NUM_EXPERTS, group)
    assert torch.equal(dispatch_out, ref_dispatch_out), f"elastic_xpu.dispatch mismatch in rank {rank}"
    expected_recv_topk_idx = torch.full(
        (dispatch_out.shape[0], topk_idx.shape[1]),
        -1,
        device=device,
        dtype=topk_idx.dtype,
    )
    expected_recv_topk_weights = torch.zeros(
        (dispatch_out.shape[0], topk_idx.shape[1]),
        device=device,
        dtype=topk_weights.dtype,
    )
    flat_scatter = scatter_idx.reshape(-1).to(torch.int64)
    flat_topk_idx = topk_idx.reshape(-1)
    flat_topk_weights = topk_weights.reshape(-1)
    flat_k = torch.arange(topk_idx.shape[1], device=device, dtype=torch.int64).view(1, -1).expand(topk_idx.shape[0], -1).reshape(-1)
    valid = (flat_scatter >= 0) & (flat_scatter < expected_recv_topk_idx.shape[0])
    expected_recv_topk_idx[flat_scatter[valid], flat_k[valid]] = flat_topk_idx[valid]
    expected_recv_topk_weights[flat_scatter[valid], flat_k[valid]] = flat_topk_weights[valid]

    assert torch.equal(recv_topk_idx, expected_recv_topk_idx)
    assert torch.allclose(recv_topk_weights, expected_recv_topk_weights)
    assert handle is not None
    assert handle.scatter_idx is not None
    assert torch.equal(handle.scatter_idx, scatter_idx)

    # Combine should match the deepep reference path.
    rank_output_ptrs = build_combine_rank_output_ptrs(expert_output_local, topk_idx, group=group)
    combine_out = torch.zeros(TOKENS_PER_RANK, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    ref_combine_out = torch.zeros_like(combine_out)

    for _ in range(WARMUP):
        combine_out.zero_()
        combine_out, returned_topk_weights, _ = buffer.combine(
            expert_output_local,
            handle=handle,
            topk_weights=topk_weights,
        )
    torch.xpu.synchronize()
    dist.barrier()

    for _ in range(LOOP):
        combine_out.zero_()
        combine_out, returned_topk_weights, _ = buffer.combine(
            expert_output_local,
            handle=handle,
            topk_weights=topk_weights,
        )
    torch.xpu.synchronize()
    dist.barrier()

    deepep_owner_combine(
        expert_output=expert_output_local,
        topk_idx=topk_idx,
        scatter_idx=scatter_idx,
        topk_weights=topk_weights,
        output=ref_combine_out,
        num_experts=NUM_EXPERTS,
        group=group,
        rank_output_ptrs=rank_output_ptrs,
    )
    assert torch.allclose(combine_out, ref_combine_out, atol=1e-2, rtol=1e-2), (
        f"elastic_xpu.combine mismatch in rank {rank}"
    )
    assert torch.allclose(returned_topk_weights, topk_weights)

    if rank == 0:
        print("[Summary] elastic_xpu.dispatch/combine correctness PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    check_elastic_xpu_dispatch_and_combine()
