"""
Performance benchmark for SymmBuffer fusion APIs on XPU.

Benchmarks two APIs and compares against their standalone counterparts:
1. SymmBuffer.allgather_local_permute_fusion  vs  allgather_local_permute_fusion()
2. SymmBuffer.unpermute_reducescatter_fusion   vs  unpermute_reducescatter()

Usage:
    mpirun -n 2 python test_elastic_xpu_fusion_perf.py
"""

import os

import torch
import torch.distributed as dist

import env

from allgather_local_permute_fusion import (
    allgather_local_permute_fusion,
    build_allgather_rank_buffers_ptr,
    compute_scatter_idx,
)
from unpermute_reducescatter_fusion import (
    unpermute_reducescatter,
    unpermute_allreduce_simple,
)
from symm_buffer import SymmBuffer

TOKENS_PER_RANK = env.tokens_per_rank()
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
LOOP = 40
WARMUP = 20
PCIE_DISCOUNT = 0.7
CROSS_GPU_BW_GBPS = 31.5 * PCIE_DISCOUNT
HBM_BW_GBPS = 437.0


def build_allgather_local_permute_reference(hidden_shard, scatter_idx, group):
    world_size = dist.get_world_size(group)
    num_tokens_per_rank, hidden = hidden_shard.shape
    num_tokens, topk = scatter_idx.shape
    gathered = [torch.empty_like(hidden_shard) for _ in range(world_size)]
    dist.all_gather(gathered, hidden_shard, group=group)
    out = torch.empty(
        num_tokens * topk, hidden, device=hidden_shard.device, dtype=hidden_shard.dtype
    )
    for src_rank in range(world_size):
        token_offset = src_rank * num_tokens_per_rank
        torch.ops.symm_mem.local_permute_copy_(
            gathered[src_rank], scatter_idx, token_offset, out
        )
    return out


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 * 1024)


def project_time_ms(bytes_count, bw_gbps):
    return bytes_count / (bw_gbps * 1e9) * 1e3


def init_distributed():
    env.setup_distributed_env(master_addr="localhost", master_port="29525")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def make_inputs(rank, world_size, device):
    num_tokens_per_rank = TOKENS_PER_RANK
    num_tokens = num_tokens_per_rank * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        num_tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )

    torch.manual_seed(42)
    global_topk_idx = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOPK), device=device, dtype=torch.int32
    )
    topk_idx = global_topk_idx[
        rank * num_tokens_per_rank : (rank + 1) * num_tokens_per_rank
    ].clone()

    torch.manual_seed(777)
    global_topk_weights = torch.rand(
        num_tokens, TOPK, device=device, dtype=torch.float32
    )
    global_topk_weights = global_topk_weights / global_topk_weights.sum(
        dim=1, keepdim=True
    )
    topk_weights = global_topk_weights[
        rank * num_tokens_per_rank : (rank + 1) * num_tokens_per_rank
    ].clone()

    return hidden_shard, topk_idx, global_topk_idx, topk_weights


def timed_loop(fn, loop, warmup):
    """Run fn in a warmup+timed loop, return per-iteration latencies (ms)."""
    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    for i in range(loop):
        if i >= warmup:
            begin_events[i].record()
        fn()
        if i >= warmup:
            end_events[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    return [begin_events[i].elapsed_time(end_events[i]) for i in range(warmup, loop)]


def print_latency_summary(label, latencies, rank):
    avg = sum(latencies) / len(latencies)
    print(f"[{label} latency rank {rank}] {[f'{l:.3f}' for l in latencies]} ms")
    return avg


# ---------------------------------------------------------------------------
# Allgather + local permute fusion benchmark
# ---------------------------------------------------------------------------

def bench_allgather_permute_fusion(sbuf, hidden_shard, topk_idx, topk_weights,
                                    global_topk_idx, rank, world_size, device):
    num_tokens_per_rank = TOKENS_PER_RANK
    num_tokens = num_tokens_per_rank * world_size
    group = dist.group.WORLD

    scatter_idx, _ = compute_scatter_idx(global_topk_idx, num_experts=NUM_EXPERTS)
    backend_stream = torch.xpu.Stream()

    # --- Standalone API (run first to avoid workspace pointer invalidation) ---
    rank_buffers = build_allgather_rank_buffers_ptr(hidden_shard, group=group)
    remap = torch.empty(
        (num_tokens * TOPK, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )

    def run_standalone():
        allgather_local_permute_fusion(
            hidden_shard=hidden_shard,
            topk_idx=global_topk_idx,
            scatter_idx=scatter_idx,
            remap_hidden_states=remap,
            group=group,
            backend_stream=backend_stream,
            rank_buffers_ptr=rank_buffers,
        )

    standalone_latencies = timed_loop(run_standalone, LOOP, WARMUP)

    # Pre-allocate SymmBuffer outputs
    remap_symm = torch.empty(
        (num_tokens * TOPK, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )

    # --- SymmBuffer API ---
    symm_handle = None

    def run_symm():
        nonlocal symm_handle
        _, symm_handle = sbuf.allgather_local_permute_fusion(
            hidden_shard=hidden_shard,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=NUM_EXPERTS,
            remap_hidden_states=remap_symm,
        )

    symm_latencies = timed_loop(run_symm, LOOP, WARMUP)

    # Per-op breakdown (notify_dispatch vs ring op) when enabled.
    sbuf.report_ring_timing(rank)

    # --- Accuracy check: SymmBuffer vs all_gather reference (matches dist UT) ---
    run_symm()
    torch.xpu.synchronize()
    expert_cumsum = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.int32)
    expert_cumsum[1:] = symm_handle.rows_per_expert[:-1]
    expert_cumsum = expert_cumsum.cumsum(0).to(torch.int32)
    abs_scatter_idx = (
        expert_cumsum[symm_handle.global_topk_idx] + symm_handle.scatter_idx
    ).to(torch.int32)
    ref_dispatch = build_allgather_local_permute_reference(
        hidden_shard, abs_scatter_idx, group
    )
    dispatch_match = torch.equal(remap_symm, ref_dispatch)
    print(
        f"[Rank {rank}] allgather_permute accuracy: "
        f"match={dispatch_match}"
        + (
            f" max_diff={(remap_symm - ref_dispatch).abs().max().item():.6f}"
            if not dispatch_match
            else ""
        )
    )
    assert dispatch_match, (
        f"allgather_permute mismatch on rank {rank}: "
        f"max_diff={(remap_symm - ref_dispatch).abs().max().item():.6f}"
    )

    avg_symm = print_latency_summary(
        "SymmBuffer.allgather_permute", symm_latencies, rank
    )
    avg_standalone = print_latency_summary(
        "Standalone allgather_permute", standalone_latencies, rank
    )

    if rank == 0:
        print(f"\n{'='*70}")
        print("[Allgather + Permute Fusion]")
        print(
            f"  SymmBuffer:  avg={avg_symm:.3f} ms  "
            f"min={min(symm_latencies):.3f} ms  max={max(symm_latencies):.3f} ms"
        )
        print(
            f"  Standalone:  avg={avg_standalone:.3f} ms  "
            f"min={min(standalone_latencies):.3f} ms  max={max(standalone_latencies):.3f} ms"
        )
        if avg_standalone > 0:
            overhead = (avg_symm - avg_standalone) / avg_standalone * 100
            print(
                f"  SymmBuffer overhead: {overhead:+.1f}%"
            )

        elem_size = hidden_shard.element_size()
        allgather_bytes = (world_size - 1) * num_tokens_per_rank * HIDDEN_SIZE * elem_size
        permute_write_bytes = num_tokens * TOPK * HIDDEN_SIZE * elem_size
        permute_read_bytes = num_tokens * HIDDEN_SIZE * elem_size

        proj_ag_ms = project_time_ms(allgather_bytes, CROSS_GPU_BW_GBPS)
        proj_perm_ms = project_time_ms(permute_read_bytes + permute_write_bytes, HBM_BW_GBPS)

        print(
            f"\n  [Projection] allgather={bytes_to_mb(allgather_bytes):.1f} MB "
            f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_ag_ms:.3f} ms"
        )
        print(
            f"  [Projection] permute (R+W)={bytes_to_mb(permute_read_bytes + permute_write_bytes):.1f} MB "
            f"@{HBM_BW_GBPS:.1f} GB/s -> {proj_perm_ms:.3f} ms"
        )
        lower_bound = proj_ag_ms + proj_perm_ms
        print(f"  [Projection] lower_bound={lower_bound:.3f} ms")
        print(
            f"  [Efficiency] standalone={lower_bound / avg_standalone * 100:.1f}%  "
            f"symmbuffer={lower_bound / avg_symm * 100:.1f}%"
        )
        print(f"{'='*70}\n")

    return abs_scatter_idx, symm_handle


# ---------------------------------------------------------------------------
# Unpermute + reduce-scatter fusion benchmark
# ---------------------------------------------------------------------------

def bench_unpermute_reducescatter_fusion(sbuf, scatter_idx, global_topk_idx,
                                         global_topk_weights, symm_handle,
                                         rank, world_size, device):
    num_tokens_per_rank = TOKENS_PER_RANK
    num_tokens = num_tokens_per_rank * world_size
    group = dist.group.WORLD

    torch.manual_seed(2026)
    expert_output = torch.randn(
        num_tokens * TOPK, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )

    backend_stream = torch.xpu.Stream()

    # --- Standalone API (run first to avoid workspace pointer invalidation) ---
    output_standalone = torch.zeros(
        num_tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )

    def run_standalone():
        unpermute_reducescatter(
            expert_output=expert_output,
            scatter_idx=scatter_idx,
            topk_weights=global_topk_weights,
            output=output_standalone,
            group=group,
            backend_stream=backend_stream,
        )

    standalone_latencies = timed_loop(run_standalone, LOOP, WARMUP)

    # --- SymmBuffer API ---
    output_symm = torch.zeros(
        num_tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )

    def run_symm():
        sbuf.unpermute_reducescatter_fusion(
            expert_output=expert_output,
            handle=symm_handle,
            output=output_symm,
        )

    symm_latencies = timed_loop(run_symm, LOOP, WARMUP)

    # --- Accuracy check: SymmBuffer vs all_gather reference (matches dist UT) ---
    output_symm.zero_()
    run_symm()
    ref_reduce_out = torch.zeros_like(output_symm)
    unpermute_allreduce_simple(
        expert_output=expert_output,
        scatter_idx=scatter_idx,
        topk_weights=global_topk_weights,
        output=ref_reduce_out,
        group=group,
    )
    torch.xpu.synchronize()
    atol = 0.025 * world_size + 0.01
    unperm_match = torch.allclose(output_symm, ref_reduce_out, atol=atol, rtol=1e-2)
    max_diff = (output_symm - ref_reduce_out).abs().max().item()
    print(
        f"[Rank {rank}] unpermute_reducescatter accuracy: "
        f"match={unperm_match} max_diff={max_diff:.6f} atol={atol}"
    )
    assert unperm_match, (
        f"unpermute_reducescatter mismatch on rank {rank}: "
        f"max_diff={max_diff:.6f}, atol={atol}"
    )

    avg_symm = print_latency_summary(
        "SymmBuffer.unpermute_rs", symm_latencies, rank
    )
    avg_standalone = print_latency_summary(
        "Standalone unpermute_rs", standalone_latencies, rank
    )

    if rank == 0:
        print(f"\n{'='*70}")
        print("[Unpermute + Reduce-Scatter Fusion]")
        print(
            f"  SymmBuffer:  avg={avg_symm:.3f} ms  "
            f"min={min(symm_latencies):.3f} ms  max={max(symm_latencies):.3f} ms"
        )
        print(
            f"  Standalone:  avg={avg_standalone:.3f} ms  "
            f"min={min(standalone_latencies):.3f} ms  max={max(standalone_latencies):.3f} ms"
        )
        if avg_standalone > 0:
            overhead = (avg_symm - avg_standalone) / avg_standalone * 100
            print(
                f"  SymmBuffer overhead: {overhead:+.1f}%"
            )

        elem_size = expert_output.element_size()
        rs_bytes = (world_size - 1) * num_tokens_per_rank * HIDDEN_SIZE * elem_size
        unperm_read_bytes = num_tokens * TOPK * HIDDEN_SIZE * elem_size
        unperm_write_bytes = num_tokens_per_rank * HIDDEN_SIZE * elem_size

        proj_rs_ms = project_time_ms(rs_bytes, CROSS_GPU_BW_GBPS)
        proj_compute_ms = project_time_ms(
            unperm_read_bytes + unperm_write_bytes, HBM_BW_GBPS
        )

        print(
            f"\n  [Projection] reduce_scatter={bytes_to_mb(rs_bytes):.1f} MB "
            f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_rs_ms:.3f} ms"
        )
        print(
            f"  [Projection] unpermute (R+W)={bytes_to_mb(unperm_read_bytes + unperm_write_bytes):.1f} MB "
            f"@{HBM_BW_GBPS:.1f} GB/s -> {proj_compute_ms:.3f} ms"
        )
        lower_bound = proj_rs_ms + proj_compute_ms
        print(f"  [Projection] lower_bound={lower_bound:.3f} ms")
        print(
            f"  [Efficiency] standalone={lower_bound / avg_standalone * 100:.1f}%  "
            f"symmbuffer={lower_bound / avg_symm * 100:.1f}%"
        )
        print(f"{'='*70}\n")


def main():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    hidden_shard, topk_idx, global_topk_idx, topk_weights = make_inputs(
        rank, world_size, device
    )

    # Reconstruct global topk_weights for standalone API
    torch.manual_seed(777)
    global_topk_weights = torch.rand(
        TOKENS_PER_RANK * world_size, TOPK, device=device, dtype=torch.float32
    )
    global_topk_weights = global_topk_weights / global_topk_weights.sum(
        dim=1, keepdim=True
    )

    sbuf = SymmBuffer(
        group=group,
        num_max_tokens_per_rank=TOKENS_PER_RANK,
        hidden=HIDDEN_SIZE,
        num_topk=TOPK,
    )

    # Benchmark 1: allgather + permute fusion
    scatter_idx, symm_handle = bench_allgather_permute_fusion(
        sbuf, hidden_shard, topk_idx, topk_weights,
        global_topk_idx, rank, world_size, device,
    )

    torch.xpu.synchronize()
    dist.barrier()
    torch.xpu.empty_cache()

    # Benchmark 2: unpermute + reduce-scatter fusion
    bench_unpermute_reducescatter_fusion(
        sbuf, scatter_idx, global_topk_idx, global_topk_weights, symm_handle,
        rank, world_size, device,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
