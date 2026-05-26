"""
Accuracy test for SymmBuffer fusion APIs on XPU.

Validates two fusion APIs in symm_buffer.SymmBuffer:
1. allgather + local permute fusion
2. unpermute + reduce-scatter fusion

Tests both the handle-based API (permute → handle → unpermute) and the
explicit-argument API.

Usage:
    mpirun -n 2 python test_elastic_xpu_fusion_dist.py
"""

import os

import torch
import torch.distributed as dist

from symm_buffer import SymmBuffer, SymmHandle
from unpermute_reducescatter_fusion import unpermute_allreduce_simple

TOKENS_PER_RANK = 512
HIDDEN_SIZE = 512
TOPK = 4
NUM_EXPERTS = 32


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29524"
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def build_allgather_local_permute_reference(
    hidden_shard: torch.Tensor,
    scatter_idx: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    num_tokens_per_rank, hidden = hidden_shard.shape
    num_tokens, topk = scatter_idx.shape
    assert num_tokens == num_tokens_per_rank * world_size

    gathered = [torch.empty_like(hidden_shard) for _ in range(world_size)]
    dist.all_gather(gathered, hidden_shard, group=group)

    out = torch.empty(num_tokens * topk, hidden, device=hidden_shard.device, dtype=hidden_shard.dtype)
    for src_rank in range(world_size):
        token_offset = src_rank * num_tokens_per_rank
        torch.ops.symm_mem.local_permute_copy_(
            gathered[src_rank],
            scatter_idx,
            token_offset,
            out,
        )

    _ = rank  # suppress unused var warning in linters
    return out


def check_elastic_xpu_fusions():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    num_tokens = TOKENS_PER_RANK * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        TOKENS_PER_RANK,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )

    # Build deterministic global topk and per-rank local slices.
    torch.manual_seed(42)
    global_topk_idx = torch.randint(
        0,
        NUM_EXPERTS,
        (num_tokens, TOPK),
        device=device,
        dtype=torch.int32,
    )
    topk_idx = global_topk_idx[
        rank * TOKENS_PER_RANK : (rank + 1) * TOKENS_PER_RANK
    ].clone()

    torch.manual_seed(777)
    global_topk_weights = torch.rand(
        num_tokens,
        TOPK,
        device=device,
        dtype=torch.float32,
    )
    global_topk_weights = global_topk_weights / global_topk_weights.sum(dim=1, keepdim=True)
    topk_weights = global_topk_weights[
        rank * TOKENS_PER_RANK : (rank + 1) * TOKENS_PER_RANK
    ].clone()

    flat_k = torch.arange(TOPK, device=device, dtype=torch.int32).view(1, -1).expand(num_tokens, -1).reshape(-1)

    flat_k = torch.arange(TOPK, device=device, dtype=torch.int32).view(1, -1).expand(num_tokens, -1).reshape(-1)

    symm_buf = SymmBuffer(
        group=group,
        num_max_tokens_per_rank=TOKENS_PER_RANK,
        hidden=HIDDEN_SIZE,
        num_topk=TOPK,
    )
    remap_hidden_states = torch.empty(
        (num_tokens * TOPK, HIDDEN_SIZE),
        device=device,
        dtype=torch.bfloat16,
    )

    # 1) allgather + local permute fusion (returns handle)
    dispatch_out, _, _, handle = symm_buf.allgather_local_permute_fusion(
        hidden_shard=hidden_shard,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=NUM_EXPERTS,
        remap_hidden_states=remap_hidden_states,
    )

    assert isinstance(handle, SymmHandle), "Expected SymmHandle from allgather_local_permute_fusion"
    assert handle.num_tokens_per_rank == TOKENS_PER_RANK

    flat_scatter = handle.scatter_idx.reshape(-1)
    rebuilt_topk_idx = handle.recv_topk_idx[flat_scatter, flat_k].reshape(num_tokens, TOPK)
    rebuilt_topk_weights = handle.recv_topk_weights[flat_scatter, flat_k].reshape(num_tokens, TOPK)
    assert torch.equal(rebuilt_topk_idx, global_topk_idx), (
        f"recv_topk_idx -> global_topk_idx mismatch on rank {rank}"
    )
    assert torch.allclose(rebuilt_topk_weights, global_topk_weights, atol=0, rtol=0), (
        f"recv_topk_weights -> global_topk_weights mismatch on rank {rank}"
    )

    ref_dispatch = build_allgather_local_permute_reference(
        hidden_shard=hidden_shard,
        scatter_idx=handle.scatter_idx,
        group=group,
    )
    assert torch.equal(dispatch_out, ref_dispatch), (
        f"allgather_local_permute_fusion mismatch on rank {rank}"
    )

    # 2) unpermute + reduce-scatter fusion (via handle)
    torch.manual_seed(2026)
    expert_output = torch.randn(
        num_tokens * TOPK,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )

    reduce_out_buf = torch.zeros(
        TOKENS_PER_RANK,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.bfloat16,
    )
    reduce_out = symm_buf.unpermute_reducescatter_fusion(
        expert_output=expert_output,
        recv_topk_idx=handle.recv_topk_idx,
        recv_topk_weights=handle.recv_topk_weights,
        handle=handle,
        output=reduce_out_buf,
    )

    ref_reduce_out = torch.zeros_like(reduce_out)
    unpermute_allreduce_simple(
        expert_output=expert_output,
        scatter_idx=handle.scatter_idx,
        topk_weights=global_topk_weights,
        output=ref_reduce_out,
        group=group,
    )

    atol = 0.025 * world_size + 0.01
    assert torch.allclose(reduce_out, ref_reduce_out, atol=atol, rtol=1e-2), (
        f"unpermute_reducescatter_fusion mismatch on rank {rank}: "
        f"max_diff={(reduce_out - ref_reduce_out).abs().max().item():.6f}, atol={atol}"
    )
    print(f"[Rank {rank}] SymmBuffer fusion UT passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    check_elastic_xpu_fusions()
