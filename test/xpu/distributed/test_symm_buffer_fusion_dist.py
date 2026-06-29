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
import time

import torch
import torch.distributed as dist

from symm_buffer import SymmBuffer, SymmHandle
from unpermute_reducescatter_fusion import unpermute_allreduce_simple

TOKENS_PER_RANK_LIST = [128, 256, 512]
MAX_TOKENS_PER_RANK = max(TOKENS_PER_RANK_LIST)
HIDDEN_SIZE = 512
TOPK = 4
NUM_EXPERTS = 32
WARMUP_ITERS = 3
PERF_ITERS = 10


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

    # Init SymmBuffer once with the max capacity
    symm_buf = SymmBuffer(
        group=group,
        num_max_tokens_per_rank=MAX_TOKENS_PER_RANK,
        hidden=HIDDEN_SIZE,
        num_topk=TOPK,
    )

    for tokens_per_rank in TOKENS_PER_RANK_LIST:
        print(f"[Rank {rank}] Testing with TOKENS_PER_RANK={tokens_per_rank}")
        num_tokens = tokens_per_rank * world_size

        torch.manual_seed(1234 + rank)
        hidden_shard = torch.randn(
            tokens_per_rank,
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
            rank * tokens_per_rank : (rank + 1) * tokens_per_rank
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
            rank * tokens_per_rank : (rank + 1) * tokens_per_rank
        ].clone()

        remap_hidden_states = torch.empty(
            (num_tokens * TOPK, HIDDEN_SIZE),
            device=device,
            dtype=torch.bfloat16,
        )

        # 1) allgather + local permute fusion (returns handle)
        dispatch_out, handle = symm_buf.allgather_local_permute_fusion(
            hidden_shard=hidden_shard,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=NUM_EXPERTS,
            remap_hidden_states=remap_hidden_states,
        )

        assert isinstance(handle, SymmHandle), "Expected SymmHandle from allgather_local_permute_fusion"
        assert handle.num_tokens_per_rank == tokens_per_rank

        # Compute absolute scatter_idx from relative + rows_per_expert
        expert_cumsum = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.int32)
        expert_cumsum[1:] = handle.rows_per_expert[:-1]
        expert_cumsum = expert_cumsum.cumsum(0).to(torch.int32)
        abs_scatter_idx = (expert_cumsum[handle.global_topk_idx] + handle.scatter_idx).to(torch.int32)

        # Verify handle.global_topk_weights matches the global weights
        assert torch.allclose(handle.global_topk_weights, global_topk_weights, atol=0, rtol=0), (
            f"handle.global_topk_weights mismatch on rank {rank}"
        )

        ref_dispatch = build_allgather_local_permute_reference(
            hidden_shard=hidden_shard,
            scatter_idx=abs_scatter_idx,
            group=group,
        )
        assert torch.equal(dispatch_out, ref_dispatch), (
            f"allgather_local_permute_fusion mismatch on rank {rank}"
        )

        # V2 validation: check scatter_idx is expert-relative
        assert handle.global_topk_idx is not None, "Handle must have global_topk_idx"
        assert handle.rows_per_expert is not None, "Handle must have rows_per_expert"

        # Verify scatter_idx values are within [0, rows_per_expert[expert_id]) for each entry
        for t in range(min(num_tokens, 8)):  # spot-check first 8 tokens
            for k in range(TOPK):
                expert_id = handle.global_topk_idx[t, k].item()
                rel_pos = handle.scatter_idx[t, k].item()
                expert_count = handle.rows_per_expert[expert_id].item()
                assert 0 <= rel_pos < expert_count, (
                    f"scatter_idx out of range: token={t}, k={k}, "
                    f"expert={expert_id}, rel_pos={rel_pos}, count={expert_count}"
                )

        # Verify rows_per_expert exactly matches routing histogram from global_topk_idx.
        expected_rows_per_expert = torch.bincount(
            handle.global_topk_idx.reshape(-1).to("cpu", dtype=torch.int64),
            minlength=NUM_EXPERTS,
        ).to(device=device, dtype=torch.int32)
        assert torch.equal(handle.rows_per_expert, expected_rows_per_expert), (
            f"rows_per_expert mismatch on rank {rank}: "
            f"max_abs_diff={(handle.rows_per_expert - expected_rows_per_expert).abs().max().item()}"
        )

        # Verify rows_per_expert sums to num_tokens * topk
        total_assigned = handle.rows_per_expert.sum().item()
        assert total_assigned == num_tokens * TOPK, (
            f"rows_per_expert sum ({total_assigned}) != num_tokens * topk ({num_tokens * TOPK})"
        )
        print(f"[Rank {rank}] V2 validations passed (tokens_per_rank={tokens_per_rank})")

        # 2) unpermute + reduce-scatter fusion (via handle)
        torch.manual_seed(2026)
        expert_output = torch.randn(
            num_tokens * TOPK,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )

        reduce_out_buf = torch.zeros(
            tokens_per_rank,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.bfloat16,
        )
        reduce_out = symm_buf.unpermute_reducescatter_fusion(
            expert_output=expert_output,
            handle=handle,
            output=reduce_out_buf,
        )

        ref_reduce_out = torch.zeros_like(reduce_out)
        unpermute_allreduce_simple(
            expert_output=expert_output,
            scatter_idx=abs_scatter_idx,
            topk_weights=global_topk_weights,
            output=ref_reduce_out,
            group=group,
        )

        atol = 0.025 * world_size + 0.01
        assert torch.allclose(reduce_out, ref_reduce_out, atol=atol, rtol=1e-2), (
            f"unpermute_reducescatter_fusion mismatch on rank {rank}: "
            f"max_diff={(reduce_out - ref_reduce_out).abs().max().item():.6f}, atol={atol}"
        )
        print(f"[Rank {rank}] SymmBuffer fusion UT passed (tokens_per_rank={tokens_per_rank})")

        # --- Performance measurement ---
        # Warmup
        for _ in range(WARMUP_ITERS):
            symm_buf.allgather_local_permute_fusion(
                hidden_shard=hidden_shard,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=NUM_EXPERTS,
                remap_hidden_states=remap_hidden_states,
            )
            reduce_out_buf.zero_()
            symm_buf.unpermute_reducescatter_fusion(
                expert_output=expert_output,
                handle=handle,
                output=reduce_out_buf,
            )
        torch.xpu.synchronize()

        # Timed: allgather + local permute
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(PERF_ITERS):
            symm_buf.allgather_local_permute_fusion(
                hidden_shard=hidden_shard,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=NUM_EXPERTS,
                remap_hidden_states=remap_hidden_states,
            )
        torch.xpu.synchronize()
        ag_time_ms = (time.perf_counter() - t0) / PERF_ITERS * 1000

        # Timed: unpermute + reduce-scatter
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        for _ in range(PERF_ITERS):
            reduce_out_buf.zero_()
            symm_buf.unpermute_reducescatter_fusion(
                expert_output=expert_output,
                handle=handle,
                output=reduce_out_buf,
            )
        torch.xpu.synchronize()
        rs_time_ms = (time.perf_counter() - t0) / PERF_ITERS * 1000

        if rank == 0:
            print(
                f"  [Perf] tokens_per_rank={tokens_per_rank:4d}  "
                f"allgather_permute={ag_time_ms:.3f}ms  "
                f"unpermute_reducescatter={rs_time_ms:.3f}ms"
            )

    print(f"[Rank {rank}] All shape variants passed!")
    dist.destroy_process_group()


if __name__ == "__main__":
    check_elastic_xpu_fusions()
