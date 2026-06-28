"""
Performance benchmark for ring_reduce_scatter_unpermute (fused combine:
symmetric-memory single-kernel MoE unpermute + ring reduce-scatter).

Compares the on-device fused ring_reduce_scatter_unpermute against the same
unfused reference used by test_unpermute_reducescatter_fusion_dist.py: the
native local_unpermute_copy_ (weighted gather, expert-centric -> token-centric)
followed by the framework collective dist.reduce_scatter_tensor. Reports a
roofline projection based on the cross-GPU link bandwidth.

Usage:
    mpirun -n 2 python benchmark_ring_reduce_scatter_unpermute_dist.py
    mpirun -n 4 python benchmark_ring_reduce_scatter_unpermute_dist.py
"""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from ring_collectives import (
    _HAS_RING_REDUCE_SCATTER_UNPERMUTE,
    build_ring_reduce_scatter_unpermute_resources,
    ring_reduce_scatter_unpermute,
)

# Importing this module loads the native local_unpermute_copy_ op used by the
# reference path (same op as test_unpermute_reducescatter_fusion_dist.py).
from unpermute_reducescatter_fusion import _HAS_LOCAL_UNPERMUTE_KERNEL

TOKENS_PER_RANK = 2048
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
DTYPE = torch.bfloat16

LOOP = 40
WARMUP = 20
ENABLE_PROFILE = False
ENABLE_PROJECTION = True

PCIE_DISCOUNT = 0.85
CROSS_GPU_BW_GBPS = 31.5 * PCIE_DISCOUNT
HBM_BW_GBPS = 437.0


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 * 1024)


def project_time_ms(bytes_count, bw_gbps):
    # GB/s is interpreted as 1e9 bytes/s for bandwidth projection.
    return bytes_count / (bw_gbps * 1e9) * 1e3


def init_distributed():
    os.environ.setdefault("RANK", str(os.environ.get("PMI_RANK", 0)))
    os.environ.setdefault("WORLD_SIZE", str(os.environ.get("PMI_SIZE", 1)))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29805")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def _owners(num_experts, world_size):
    """Contiguous-block owner assignment (matches the native kernel)."""
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


def build_routing(num_tokens, topk, num_experts, world_size, device):
    """Build global routing tables (identical on every rank): topk_idx,
    scatter_idx (owner-local compacted destination rows), topk_weights and the
    per-rank row counts (remap_rows)."""
    g = torch.Generator().manual_seed(123)
    topk_idx = torch.randint(
        0, num_experts, (num_tokens, topk), generator=g, dtype=torch.int32
    )
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


def _expert_output(owner_rank, rows, hidden, device):
    g = torch.Generator().manual_seed(1000 + owner_rank)
    if rows == 0:
        return torch.zeros(0, hidden, dtype=DTYPE, device=device)
    return torch.randn(rows, hidden, generator=g, dtype=torch.float32).to(device).to(DTYPE)


def benchmark_ring_reduce_scatter_unpermute():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    if not _HAS_RING_REDUCE_SCATTER_UNPERMUTE:
        if rank == 0:
            print("Native ring_reduce_scatter_unpermute kernel not built; run csrc/build.py first.")
        dist.destroy_process_group()
        return
    if not _HAS_LOCAL_UNPERMUTE_KERNEL:
        if rank == 0:
            print("Native local_unpermute_copy_ kernel not built; run csrc/build.py first.")
        dist.destroy_process_group()
        return

    num_tokens_per_rank = TOKENS_PER_RANK
    hidden = HIDDEN_SIZE
    topk = TOPK
    num_experts = NUM_EXPERTS
    num_tokens = num_tokens_per_rank * world_size

    topk_idx, scatter_idx, topk_weights, remap_rows, owners = build_routing(
        num_tokens, topk, num_experts, world_size, device
    )
    remap_rows_local = remap_rows[rank]

    # This rank's expert outputs (owner-local compacted rows) consumed by the
    # fused ring kernel.
    expert_output = _expert_output(rank, remap_rows_local, hidden, device)

    elem_size = expert_output.element_size()

    # ----------------------------------------------------------------------
    # Reference path: kept BYTE-FOR-BYTE consistent with
    # test_unpermute_reducescatter_fusion_dist.py so the reported reference
    # timing matches that UT.  It uses the same inputs (a full
    # [num_tokens*topk, hidden] expert_output replicated on every rank, random
    # scatter_idx, softmax-normalised weights), the same native ops
    # (local_unpermute_copy_ + dist.reduce_scatter_tensor) and the same
    # per-iteration buffer allocation.
    # ----------------------------------------------------------------------
    num_rows = num_tokens * topk
    torch.manual_seed(42)
    ref_expert_output = torch.randn(num_rows, hidden, device=device, dtype=DTYPE)
    ref_scatter_idx = torch.randint(
        0, num_rows, (num_tokens, topk), device=device, dtype=torch.int32
    )
    raw_w = torch.rand(num_tokens, topk, device=device, dtype=torch.float32)
    ref_topk_weights = raw_w / raw_w.sum(dim=1, keepdim=True)

    def run_reference():
        # Identical to run_reference_unpermute_reducescatter in
        # test_unpermute_reducescatter_fusion_dist.py.
        full_result = torch.empty(num_tokens, hidden, device=device, dtype=DTYPE)
        torch.ops.symm_mem.local_unpermute_copy_(
            ref_expert_output, ref_scatter_idx, ref_topk_weights,
            0, num_tokens, full_result,
        )
        out = torch.zeros(num_tokens_per_rank, hidden, device=device, dtype=DTYPE)
        dist.reduce_scatter_tensor(out, full_result, group=group)
        return out

    # ----------------------------------------------------------------------
    # Owner-based reference (NOT timed): reproduces the fused EP-dispatch combine
    # so the ring kernel's output can be validated.  Each rank contributes only
    # the (token, k) slots whose expert it owns (non-owned slots zero-weighted,
    # row clamped to 0), and the reduce_scatter sum across ranks yields the
    # combine; this is the same local_unpermute_copy_ + reduce_scatter recipe but
    # on the EP-dispatched expert_output.
    # ----------------------------------------------------------------------
    owner_of_slot = owners[topk_idx.reshape(-1).long()]  # [N*topk]
    owned = owner_of_slot == rank
    scatter_ref_flat = scatter_idx.reshape(-1).clone()
    scatter_ref_flat[~owned] = 0
    acc_scatter_idx = scatter_ref_flat.reshape(num_tokens, topk).contiguous().to(torch.int32)
    weight_ref_flat = topk_weights.reshape(-1).clone()
    weight_ref_flat[~owned] = 0.0
    acc_topk_weights = weight_ref_flat.reshape(num_tokens, topk).contiguous().to(torch.float32)

    def compute_fused_reference():
        full_result = torch.empty(num_tokens, hidden, device=device, dtype=DTYPE)
        torch.ops.symm_mem.local_unpermute_copy_(
            expert_output, acc_scatter_idx, acc_topk_weights, 0, num_tokens, full_result
        )
        out = torch.zeros(num_tokens_per_rank, hidden, device=device, dtype=DTYPE)
        dist.reduce_scatter_tensor(out, full_result, group=group)
        return out

    # Hoist the fused kernel's symmetric workspace / pointer setup and output
    # buffer out of the timed loop so only the device-side work is measured.
    resources = build_ring_reduce_scatter_unpermute_resources(
        expert_output, num_tokens_per_rank, group=group
    )
    fused_output = torch.empty(num_tokens_per_rank, hidden, device=device, dtype=DTYPE)

    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    ref_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    ref_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    if ENABLE_PROFILE:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()

    with prof:
        # Warm up fused ring_reduce_scatter_unpermute.
        for _ in range(WARMUP):
            out = ring_reduce_scatter_unpermute(
                expert_output, topk_idx, scatter_idx, topk_weights, num_experts,
                num_tokens_per_rank, group=group, resources=resources,
                output=fused_output,
            )
        torch.xpu.synchronize()
        dist.barrier()

        # Timed fused path.
        for i in range(LOOP):
            if i >= WARMUP:
                begin_events[i].record()
            out = ring_reduce_scatter_unpermute(
                expert_output, topk_idx, scatter_idx, topk_weights, num_experts,
                num_tokens_per_rank, group=group, resources=resources,
                output=fused_output,
            )
            if i >= WARMUP:
                end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

        # Warm up unfused reference (local_unpermute_copy_ + reduce_scatter).
        for _ in range(WARMUP):
            run_reference()
        torch.xpu.synchronize()
        dist.barrier()

        # Timed reference path.
        for i in range(LOOP):
            if i >= WARMUP:
                ref_begin_events[i].record()
            run_reference()
            if i >= WARMUP:
                ref_end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    ref_latencies = [
        ref_begin_events[i].elapsed_time(ref_end_events[i]) for i in range(WARMUP, LOOP)
    ]

    if ENABLE_PROFILE:
        prof.export_chrome_trace(f"./profile_ring_reduce_scatter_unpermute_rank{rank}.json")

    # Accuracy check (single fresh pass): fused vs its owner-based reference.
    out = ring_reduce_scatter_unpermute(
        expert_output, topk_idx, scatter_idx, topk_weights, num_experts,
        num_tokens_per_rank, group=group, resources=resources,
    ).clone()
    ref = compute_fused_reference().clone()
    torch.xpu.synchronize()
    max_err = (out.float() - ref.float()).abs().max().item()
    tol = 1e-2 * ref.float().abs().max().clamp_min(1.0).item()
    ok = max_err <= tol

    print(f"[Ring reduce_scatter_unpermute time in rank {rank}] {latencies} ms")
    print(f"[Reference unpermute+reduce_scatter time in rank {rank}] {ref_latencies} ms")

    if rank == 0:
        avg_ring = sum(latencies) / len(latencies)
        avg_ref = sum(ref_latencies) / len(ref_latencies)
        print(f"[Accuracy] ring_reduce_scatter_unpermute match={ok} max_err={max_err} tol={tol}")
        print(
            f"[Summary] avg_fused={avg_ring:.3f} ms, "
            f"avg_reference={avg_ref:.3f} ms, speedup={avg_ref / avg_ring:.3f}x"
        )

        if ENABLE_PROJECTION:
            # Per-rank cross-GPU payload moved across the ring reduce-scatter:
            # (world_size - 1) token blocks.
            rs_bytes = (world_size - 1) * num_tokens_per_rank * hidden * elem_size
            # Unpermute traffic: read num_tokens*topk hidden vectors, write
            # num_tokens hidden vectors (the per-rank token-centric result).
            read_bytes = num_tokens * topk * hidden * elem_size
            write_bytes = num_tokens * hidden * elem_size

            proj_rs_ms = project_time_ms(rs_bytes, CROSS_GPU_BW_GBPS)
            proj_compute_ms = project_time_ms(read_bytes + write_bytes, HBM_BW_GBPS)
            proj_lower_bound = proj_rs_ms + proj_compute_ms

            print(
                f"[Projection] reduce_scatter_bytes={bytes_to_mb(rs_bytes):.2f} MB "
                f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_rs_ms:.3f} ms"
            )
            print(
                f"[Projection] unpermute_compute (read+write)={bytes_to_mb(read_bytes + write_bytes):.2f} MB "
                f"@{HBM_BW_GBPS:.1f} GB/s -> {proj_compute_ms:.3f} ms"
            )
            print(f"[Projection] fused_lower_bound={proj_lower_bound:.3f} ms")
            print(
                f"[Gap Analysis] actual={avg_ring:.3f} ms vs projected={proj_lower_bound:.3f} ms, "
                f"ratio={avg_ring / proj_lower_bound:.2f}x, "
                f"efficiency={proj_lower_bound / avg_ring * 100:.1f}%"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    benchmark_ring_reduce_scatter_unpermute()
