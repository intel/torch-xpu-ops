"""Accuracy test for FP8 allgather + permute with per-token scale on XPU.

Validates the FP8 dispatch path of ``SymmBuffer.allgather_local_permute_fusion``:
  1. FP8 hidden states (torch.float8_e4m3fn / torch.float8_e5m2) are allgathered
     and permuted byte-exactly into expert-centric layout.
  2. The optional per-token ``scale`` [num_tokens_per_rank] is allgathered into
     ``handle.global_scale`` [num_tokens] (via notify_dispatch_v2) and permuted
     into ``handle.permuted_scale`` [remap_rows] aligned with the dispatch output.

Usage:
    mpirun -n 2 python test_allgather_permute_fp8_scale_dist.py
"""

import os

import torch
import torch.distributed as dist

from symm_buffer import SymmBuffer, SymmHandle

TOKENS_PER_RANK_LIST = [128, 256, 512]
MAX_TOKENS_PER_RANK = max(TOKENS_PER_RANK_LIST)
HIDDEN_SIZE = 512
TOPK = 4
NUM_EXPERTS = 32
FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29525"
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def _abs_scatter_idx(handle: SymmHandle, num_experts: int, device) -> torch.Tensor:
    """Reconstruct absolute scatter_idx from the handle (matches kernel)."""
    expert_cumsum = torch.zeros(num_experts, device=device, dtype=torch.int32)
    expert_cumsum[1:] = handle.rows_per_expert[:-1]
    expert_cumsum = expert_cumsum.cumsum(0).to(torch.int32)
    return (expert_cumsum[handle.global_topk_idx] + handle.scatter_idx).to(torch.int32)


def _allgather_cat(local: torch.Tensor, group) -> torch.Tensor:
    """All-gather a per-rank tensor and concatenate along dim 0."""
    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local, group=group)
    return torch.cat(gathered, dim=0)


def check_fp8_scale_dispatch():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    for fp8_dtype in FP8_DTYPES:
        symm_buf = SymmBuffer(
            group=group,
            num_max_tokens_per_rank=MAX_TOKENS_PER_RANK,
            hidden=HIDDEN_SIZE,
            num_topk=TOPK,
            hidden_dtype=fp8_dtype,
        )

        for tokens_per_rank in TOKENS_PER_RANK_LIST:
            num_tokens = tokens_per_rank * world_size

            # FP8 hidden states (quantize a bf16 draw so the bytes are valid).
            torch.manual_seed(1234 + rank)
            hidden_bf16 = torch.randn(
                tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
            )
            hidden_shard = hidden_bf16.to(fp8_dtype)

            # Per-token scale [tokens_per_rank].
            torch.manual_seed(99 + rank)
            scale = torch.rand(tokens_per_rank, device=device, dtype=torch.float32) + 0.5

            # Deterministic global routing split across ranks.
            torch.manual_seed(42)
            global_topk_idx = torch.randint(
                0, NUM_EXPERTS, (num_tokens, TOPK), device=device, dtype=torch.int32
            )
            topk_idx = global_topk_idx[
                rank * tokens_per_rank : (rank + 1) * tokens_per_rank
            ].clone()

            torch.manual_seed(777)
            global_topk_weights = torch.rand(
                num_tokens, TOPK, device=device, dtype=torch.float32
            )
            topk_weights = global_topk_weights[
                rank * tokens_per_rank : (rank + 1) * tokens_per_rank
            ].clone()

            remap_hidden_states = torch.empty(
                (num_tokens * TOPK, HIDDEN_SIZE), device=device, dtype=fp8_dtype
            )

            dispatch_out, handle = symm_buf.allgather_local_permute_fusion(
                hidden_shard=hidden_shard,
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=NUM_EXPERTS,
                remap_hidden_states=remap_hidden_states,
                scale=scale,
            )

            assert isinstance(handle, SymmHandle)
            assert handle.global_scale is not None
            assert handle.permuted_scale is not None

            abs_scatter_idx = _abs_scatter_idx(handle, NUM_EXPERTS, device)
            flat_dst = abs_scatter_idx.reshape(-1)
            valid = flat_dst >= 0

            # --- Reference FP8 permute (pure byte movement). ---
            global_hidden = _allgather_cat(hidden_shard, group)  # [num_tokens, H]
            src_bytes = (
                global_hidden.view(torch.uint8)
                .repeat_interleave(TOPK, dim=0)  # [num_tokens*topk, H]
            )
            ref_out = torch.empty(
                num_tokens * TOPK, HIDDEN_SIZE, device=device, dtype=torch.uint8
            )
            ref_out[flat_dst[valid].long()] = src_bytes[valid]
            assert torch.equal(dispatch_out.view(torch.uint8), ref_out), (
                f"FP8 dispatch mismatch (dtype={fp8_dtype}, rank={rank}, "
                f"tokens_per_rank={tokens_per_rank})"
            )

            # --- Reference gathered + permuted scale. ---
            global_scale_ref = _allgather_cat(scale, group)  # [num_tokens]
            assert torch.equal(handle.global_scale, global_scale_ref), (
                f"global_scale mismatch (dtype={fp8_dtype}, rank={rank})"
            )

            permuted_scale_ref = torch.zeros(
                num_tokens * TOPK, device=device, dtype=torch.float32
            )
            token_scale = global_scale_ref.repeat_interleave(TOPK)
            permuted_scale_ref[flat_dst[valid].long()] = token_scale[valid]
            assert torch.equal(handle.permuted_scale, permuted_scale_ref), (
                f"permuted_scale mismatch (dtype={fp8_dtype}, rank={rank})"
            )

            print(
                f"[Rank {rank}] FP8 {fp8_dtype} + scale OK "
                f"(tokens_per_rank={tokens_per_rank})"
            )

    dist.barrier(group)
    if rank == 0:
        print("FP8 allgather+permute with per-token scale: ALL CHECKS PASSED")


if __name__ == "__main__":
    check_fp8_scale_dispatch()
    if dist.is_initialized():
        dist.destroy_process_group()
