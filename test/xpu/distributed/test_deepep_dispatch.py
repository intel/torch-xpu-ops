"""
Accuracy + performance check for deepep_owner_dispatch (distributed style)

Usage:
    mpirun -n 2 python test_deepep_dispatch.py
"""

import os

import torch
import torch.distributed as dist

from deepep_dispatch import deepep_owner_dispatch, get_expert_owner, build_rank_buffers_ptr
from allgather_local_permute_fusion import compute_scatter_idx

TOKENS_PER_RANK = 2048
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
LOOP = 40
WARMUP = 20
ENABLE_PROJECTION = True
PCIE_DISCOUNT = 0.7
CROSS_GPU_BW_GBPS = 31.5 * PCIE_DISCOUNT
HBM_BW_GBPS = 437.0


def build_dispatch_projection(
    topk_idx: torch.Tensor,
    num_tokens_per_rank: int,
    hidden_size: int,
    elem_size: int,
    num_experts: int,
    world_size: int,
):
    """Compute two projection times per rank:
    1. Remote fetch: PCIe read time for unique tokens from other ranks
    2. Topk write: HBM write time for owned assignments to remap_hidden_states
    """
    num_tokens, topk = topk_idx.shape
    bytes_per_token = hidden_size * elem_size

    owner_table = torch.tensor(
        [get_expert_owner(e, num_experts, world_size) for e in range(num_experts)],
        device=topk_idx.device,
        dtype=torch.int64,
    )
    owners_2d = owner_table[topk_idx]  # (num_tokens, topk)
    token_ids = torch.arange(num_tokens, device=topk_idx.device, dtype=torch.int64)
    src_ranks = token_ids // num_tokens_per_rank

    remote_fetch_ms = []
    topk_write_ms = []

    for owner_rank in range(world_size):
        # 1. Remote fetch: unique tokens from other ranks that have ≥1 owned expert
        has_owned = (owners_2d == owner_rank).any(dim=1)
        matching_src = src_ranks[has_owned]
        remote_token_count = (matching_src != owner_rank).sum().item()
        fetch_bytes = remote_token_count * bytes_per_token
        fetch_ms = fetch_bytes / (CROSS_GPU_BW_GBPS * 1e9) * 1e3
        remote_fetch_ms.append(fetch_ms)

        # 2. Topk write: total owned (token, k) assignments written to remap_hidden_states
        owned_assignments = (owners_2d == owner_rank).sum().item()
        write_bytes = owned_assignments * bytes_per_token
        write_ms = write_bytes / (HBM_BW_GBPS * 1e9) * 1e3
        topk_write_ms.append(write_ms)

    return remote_fetch_ms, topk_write_ms


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
    scatter_idx: torch.Tensor,
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
                    dst = int(scatter_idx[global_token_idx, k].item())
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

    # Compute expert-centric scatter_idx from global topk_idx
    scatter_idx, _ = compute_scatter_idx(topk_idx, num_experts=NUM_EXPERTS)

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
            scatter_idx=scatter_idx,
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
            scatter_idx=scatter_idx,
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
    ref = build_reference(all_hidden, topk_idx, scatter_idx, NUM_EXPERTS, rank, world_size)

    assert torch.equal(
        remap_hidden_states,
        ref,
    ), f"deepep_owner_dispatch mismatch in rank {rank}"

    # --- Baseline: allgather + local_permute on 1 stream (no overlap) ---
    gathered = [torch.empty_like(hidden_shard) for _ in range(world_size)]
    ref_output = torch.zeros(
        (num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype,
    )
    # Load local_permute_copy_ kernel
    _lp_lib = os.path.join(
        os.path.dirname(__file__), "..", "csrc", "liblocal_permute_copy.so"
    )
    if os.path.exists(_lp_lib):
        torch.ops.load_library(_lp_lib)
    has_lp = hasattr(torch.ops.symm_mem, "local_permute_copy_")

    if has_lp:
        # Warm up 1-stream baseline
        for _ in range(WARMUP):
            dist.all_gather(gathered, hidden_shard, group=group)
            for src_rank in range(world_size):
                torch.ops.symm_mem.local_permute_copy_(
                    gathered[src_rank], scatter_idx,
                    src_rank * num_tokens_per_rank, ref_output,
                )
        torch.xpu.synchronize()
        dist.barrier()

        # Timed 1-stream baseline
        ref_begin = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
        ref_end = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
        for i in range(LOOP):
            ref_begin[i].record()
            dist.all_gather(gathered, hidden_shard, group=group)
            for src_rank in range(world_size):
                torch.ops.symm_mem.local_permute_copy_(
                    gathered[src_rank], scatter_idx,
                    src_rank * num_tokens_per_rank, ref_output,
                )
            ref_end[i].record()
        torch.xpu.synchronize()
        dist.barrier()
        ref_latencies = [ref_begin[i].elapsed_time(ref_end[i]) for i in range(LOOP)]
        avg_ref = sum(ref_latencies) / len(ref_latencies)

        # Cross-check: for owned positions, ref_output (full permute) must match
        # remap_hidden_states (owner-filtered dispatch).
        owned_mask = (remap_hidden_states != 0).any(dim=1)
        assert torch.equal(
            remap_hidden_states[owned_mask],
            ref_output[owned_mask],
        ), f"Baseline local_permute vs deepep_owner_dispatch mismatch at owned positions in rank {rank}"
    else:
        avg_ref = None

    if rank == 0:
        avg = sum(latencies) / len(latencies)
        print(f"\n{'='*60}")
        print(f"[Summary] deepep_owner_dispatch avg={avg:.3f} ms")
        if avg_ref is not None:
            target = avg_ref / 2.0
            print(f"[Summary] allgather+permute 1-stream avg={avg_ref:.3f} ms")
            print(f"[Summary] target (half of 1-stream)={target:.3f} ms")
            print(
                f"[Summary] speedup vs 1-stream={avg_ref / avg:.2f}x "
                f"({'✓ MEETS TARGET' if avg <= target else '✗ MISSES TARGET'})"
            )
        print(f"{'='*60}\n")

        if ENABLE_PROJECTION and rank == 0:
            remote_fetch_ms, topk_write_ms = build_dispatch_projection(
                topk_idx=topk_idx,
                num_tokens_per_rank=num_tokens_per_rank,
                hidden_size=hidden_size,
                elem_size=hidden_shard.element_size(),
                num_experts=NUM_EXPERTS,
                world_size=world_size,
            )

            print(f"\n[Projection] PCIe BW={CROSS_GPU_BW_GBPS:.1f} GB/s, HBM BW={HBM_BW_GBPS:.1f} GB/s")
            for r in range(world_size):
                total_ms = remote_fetch_ms[r] + topk_write_ms[r]
                print(
                    f"[Projection][rank {r}] "
                    f"remote_fetch={remote_fetch_ms[r]:.3f} ms, "
                    f"topk_write={topk_write_ms[r]:.3f} ms, "
                    f"total={total_ms:.3f} ms"
                )
            worst = max(range(world_size), key=lambda r: remote_fetch_ms[r] + topk_write_ms[r])
            print(
                f"[Projection][Summary] worst_rank={worst}, "
                f"total={remote_fetch_ms[worst] + topk_write_ms[worst]:.3f} ms"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    check_deepep_owner_dispatch()
