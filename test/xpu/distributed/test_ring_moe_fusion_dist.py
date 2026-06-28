"""
Accuracy check for the fused ring MoE collectives (symmetric memory):
  - ring_allgather_permute        (dispatch: ring allgather + MoE permute)
  - ring_reduce_scatter_unpermute (combine: MoE unpermute + ring reduce-scatter)

Both use the owner-based TP+EP routing of EpDispatch / EpCombine.  All ranks
share the same global routing tables and the per-rank expert buffers are
generated deterministically, so every rank can compute the reference locally
without extra communication.

Usage:
    mpirun -n 2 python test_ring_moe_fusion_dist.py
    mpirun -n 4 python test_ring_moe_fusion_dist.py
"""
import os

import torch
import torch.distributed as dist

from ring_collectives import (
    _HAS_RING_ALLGATHER_PERMUTE,
    _HAS_RING_REDUCE_SCATTER_UNPERMUTE,
    ring_allgather_permute,
    ring_reduce_scatter_unpermute,
)

TOKENS_PER_RANK = 8
HIDDEN = 256
TOPK = 4
DTYPE = torch.bfloat16


def _init():
    os.environ.setdefault("RANK", str(os.environ.get("PMI_RANK", 0)))
    os.environ.setdefault("WORLD_SIZE", str(os.environ.get("PMI_SIZE", 1)))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29802")
    dist.init_process_group("xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(f"xpu:{rank}")
    return rank, world_size


def _owners(num_experts, world_size):
    base = num_experts // world_size
    rem = num_experts % world_size
    boundary = rem * (base + 1)
    owners = torch.empty(num_experts, dtype=torch.int64)
    for e in range(num_experts):
        if e < boundary:
            owners[e] = e // (base + 1)
        else:
            owners[e] = rem + (e - boundary) // base
    return owners


def _build_routing(num_tokens, topk, num_experts, world_size, device):
    """Build global routing tables (identical on every rank).

    Returns topk_idx [N, topk] int32, scatter_idx [N, topk] int32 (owner-local
    compacted rows), topk_weights [N, topk] float32, and remap_rows (per-rank
    row counts), with owners as an int64 [num_experts] table.
    """
    g = torch.Generator().manual_seed(123)
    topk_idx = torch.randint(0, num_experts, (num_tokens, topk), generator=g, dtype=torch.int32)
    topk_weights = torch.rand(num_tokens, topk, generator=g, dtype=torch.float32)

    owners = _owners(num_experts, world_size)
    flat_expert = topk_idx.reshape(-1).to(torch.int64)
    flat_owner = owners[flat_expert]

    scatter_flat = torch.empty(num_tokens * topk, dtype=torch.int32)
    counts = [0] * world_size
    for i in range(num_tokens * topk):
        o = int(flat_owner[i])
        scatter_flat[i] = counts[o]
        counts[o] += 1
    scatter_idx = scatter_flat.reshape(num_tokens, topk)

    return (
        topk_idx.to(device),
        scatter_idx.to(device),
        topk_weights.to(device),
        counts,
        owners.to(device),
    )


def _full_tokens(num_tokens, hidden, device):
    g = torch.Generator().manual_seed(7)
    return torch.randn(num_tokens, hidden, generator=g, dtype=torch.float32).to(device).to(DTYPE)


def _expert_output(owner_rank, rows, hidden, device):
    g = torch.Generator().manual_seed(1000 + owner_rank)
    if rows == 0:
        return torch.zeros(0, hidden, dtype=DTYPE, device=device)
    return torch.randn(rows, hidden, generator=g, dtype=torch.float32).to(device).to(DTYPE)


def test_ring_allgather_permute(rank, world_size, device):
    num_experts = world_size * 4
    T = TOKENS_PER_RANK
    N = world_size * T

    topk_idx, scatter_idx, _, remap_rows, owners = _build_routing(
        N, TOPK, num_experts, world_size, device
    )
    full_tokens = _full_tokens(N, HIDDEN, device)
    shard = full_tokens[rank * T : (rank + 1) * T].contiguous()

    out = ring_allgather_permute(
        shard, topk_idx, scatter_idx, num_experts, remap_rows[rank]
    ).clone()

    # Reference: scatter each owned (token, k) token into its compacted row.
    expected = torch.zeros(remap_rows[rank], HIDDEN, dtype=DTYPE, device=device)
    owner_of_slot = owners[topk_idx.reshape(-1).long()]  # [N*topk]
    owned = owner_of_slot == rank
    slot_gt = (torch.arange(N * TOPK, device=device) // TOPK)[owned]
    dst = scatter_idx.reshape(-1)[owned].long()
    expected[dst] = full_tokens[slot_gt]

    torch.xpu.synchronize()
    ok = torch.equal(out, expected)
    max_err = (out.float() - expected.float()).abs().max().item()
    if rank == 0:
        print(f"[ring_allgather_permute] match={ok} max_err={max_err} rows={remap_rows}")
    return ok


def test_ring_reduce_scatter_unpermute(rank, world_size, device):
    num_experts = world_size * 4
    T = TOKENS_PER_RANK
    N = world_size * T

    topk_idx, scatter_idx, topk_weights, remap_rows, owners = _build_routing(
        N, TOPK, num_experts, world_size, device
    )
    expert_out_all = [
        _expert_output(o, remap_rows[o], HIDDEN, device) for o in range(world_size)
    ]

    out = ring_reduce_scatter_unpermute(
        expert_out_all[rank], topk_idx, scatter_idx, topk_weights, num_experts, T
    ).clone()

    # Reference: combine = weighted gather summed across owners, then take this
    # rank's token block.
    out_full = torch.zeros(N, HIDDEN, dtype=torch.float32, device=device)
    owner_of_slot = owners[topk_idx.reshape(-1).long()]  # [N*topk]
    slot_gt = torch.arange(N * TOPK, device=device) // TOPK
    scatter_flat = scatter_idx.reshape(-1).long()
    weight_flat = topk_weights.reshape(-1)
    for o in range(world_size):
        mask = owner_of_slot == o
        rows = scatter_flat[mask]
        gts = slot_gt[mask]
        ws = weight_flat[mask]
        contrib = ws.unsqueeze(1) * expert_out_all[o][rows].float()
        out_full.index_add_(0, gts, contrib)
    expected = out_full[rank * T : (rank + 1) * T].to(DTYPE)

    torch.xpu.synchronize()
    max_err = (out.float() - expected.float()).abs().max().item()
    tol = 1e-2 * expected.float().abs().max().clamp_min(1.0).item()
    ok = max_err <= tol
    if rank == 0:
        print(
            f"[ring_reduce_scatter_unpermute] match={ok} max_err={max_err} tol={tol}"
        )
    return ok


def main():
    rank, world_size = _init()
    device = f"xpu:{rank}"

    if not (_HAS_RING_ALLGATHER_PERMUTE and _HAS_RING_REDUCE_SCATTER_UNPERMUTE):
        if rank == 0:
            print("Native fused ring kernels not built; run csrc/build.py first.")
        dist.destroy_process_group()
        return

    ap_ok = test_ring_allgather_permute(rank, world_size, device)
    ru_ok = test_ring_reduce_scatter_unpermute(rank, world_size, device)

    dist.barrier()
    if rank == 0:
        status = "PASS" if (ap_ok and ru_ok) else "FAIL"
        print(f"=== ring MoE fusion: {status} ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
