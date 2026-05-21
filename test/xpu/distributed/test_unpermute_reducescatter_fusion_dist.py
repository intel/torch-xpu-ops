"""
Accuracy + performance check for unpermute_reducescatter_fusion (distributed style)

Usage:
    mpirun -n 2 python test_unpermute_reducescatter_fusion_dist.py
"""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from unpermute_reducescatter_fusion import (
    unpermute_allreduce_simple,
    unpermute_reducescatter,
    unpermute_reducescatter_fusion,
    unpermute_reducescatter_fusion_native,
    build_unpermute_rank_buffers_ptr,
    _HAS_LOCAL_UNPERMUTE_KERNEL,
    _HAS_UNPERMUTE_RS_KERNEL,
)


TOKENS_PER_RANK = 2048
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
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
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29514'
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def make_inputs(rank, world_size, device, dtype=torch.bfloat16):
    """
    Generate consistent test tensors:
      expert_output: [num_tokens * topk, hidden]  expert-centric layout
      scatter_idx:   [num_tokens, topk] int32       maps (token_i, k) → row in expert_output
      topk_weights:  [num_tokens, topk] float32     routing weights (rows sum ≈ 1)
    """
    num_tokens = TOKENS_PER_RANK * world_size
    topk = TOPK
    hidden = HIDDEN_SIZE
    num_rows = num_tokens * topk  # expert_output rows

    # Each rank produces the same expert_output so reference is easy to compute
    torch.manual_seed(42)
    expert_output = torch.randn(num_rows, hidden, device=device, dtype=dtype)

    # scatter_idx: each (token, k) maps to a row in [0, num_rows)
    scatter_idx = torch.randint(0, num_rows, (num_tokens, topk), device=device, dtype=torch.int32)

    # topk_weights: random positive weights, softmax-normalised per token
    raw_w = torch.rand(num_tokens, topk, device=device, dtype=torch.float32)
    topk_weights = raw_w / raw_w.sum(dim=1, keepdim=True)

    return expert_output, scatter_idx, topk_weights


def check_unpermute_reducescatter_fusion():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    num_tokens_per_rank = TOKENS_PER_RANK
    hidden_size = HIDDEN_SIZE
    topk = TOPK
    num_tokens = num_tokens_per_rank * world_size

    expert_output, scatter_idx, topk_weights = make_inputs(rank, world_size, device)

    # Pre-allocate output buffers
    output_fused = torch.zeros(num_tokens_per_rank, hidden_size, device=device, dtype=expert_output.dtype)
    output_ref = torch.zeros(num_tokens_per_rank, hidden_size, device=device, dtype=expert_output.dtype)

    backend_stream = torch.xpu.Stream()

    # Precompute rank_buffers_ptr when the native kernel is available
    rank_buffers_ptr = None
    if _HAS_UNPERMUTE_RS_KERNEL:
        rank_buffers_ptr = build_unpermute_rank_buffers_ptr(expert_output, group=group)

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
        # ---------- Warm up fused path ----------
        for _ in range(WARMUP):
            out = torch.zeros_like(output_fused)
            unpermute_reducescatter(
                expert_output=expert_output,
                scatter_idx=scatter_idx,
                topk_weights=topk_weights,
                output=out,
                group=group,
                backend_stream=backend_stream,
                rank_buffers_ptr=rank_buffers_ptr,
            )
        torch.xpu.synchronize()
        dist.barrier()

        # ---------- Timed fused path ----------
        for i in range(LOOP):
            if i >= WARMUP:
                begin_events[i].record()
            out = torch.zeros_like(output_fused)
            unpermute_reducescatter(
                expert_output=expert_output,
                scatter_idx=scatter_idx,
                topk_weights=topk_weights,
                output=out,
                group=group,
                backend_stream=backend_stream,
                rank_buffers_ptr=rank_buffers_ptr,
            )
            if i >= WARMUP:
                end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()
        output_fused.copy_(out)

        # ---------- Warm up reference path ----------
        for _ in range(WARMUP):
            ref = torch.zeros_like(output_ref)
            unpermute_allreduce_simple(
                expert_output=expert_output,
                scatter_idx=scatter_idx,
                topk_weights=topk_weights,
                output=ref,
                group=group,
            )
        torch.xpu.synchronize()
        dist.barrier()

        # ---------- Timed reference path ----------
        for i in range(LOOP):
            if i >= WARMUP:
                ref_begin_events[i].record()
            ref = torch.zeros_like(output_ref)
            unpermute_allreduce_simple(
                expert_output=expert_output,
                scatter_idx=scatter_idx,
                topk_weights=topk_weights,
                output=ref,
                group=group,
            )
            if i >= WARMUP:
                ref_end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()
        output_ref.copy_(ref)

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    ref_latencies = [ref_begin_events[i].elapsed_time(ref_end_events[i]) for i in range(WARMUP, LOOP)]

    if ENABLE_PROFILE:
        prof.export_chrome_trace(f"./profile_unpermute_reducescatter_fusion_rank{rank}.json")

    print(f"[Fusion time in rank {rank}] {latencies} ms")
    print(f"[Reference unpermute+allreduce time in rank {rank}] {ref_latencies} ms")

    # ---------- Accuracy check ----------
    # Run one final fresh pass and compare fused vs reference
    check_fused = torch.zeros(num_tokens_per_rank, hidden_size, device=device, dtype=expert_output.dtype)
    check_ref = torch.zeros_like(check_fused)

    unpermute_reducescatter(
        expert_output=expert_output,
        scatter_idx=scatter_idx,
        topk_weights=topk_weights,
        output=check_fused,
        group=group,
        backend_stream=backend_stream,
        rank_buffers_ptr=rank_buffers_ptr,
    )
    unpermute_allreduce_simple(
        expert_output=expert_output,
        scatter_idx=scatter_idx,
        topk_weights=topk_weights,
        output=check_ref,
        group=group,
    )

    # Kernel uses different accumulation order than Python reference;
    # per-rank error ~0.025 compounds across world_size ranks after reduce.
    atol = 0.025 * world_size + 0.01
    assert torch.allclose(check_fused, check_ref, atol=atol, rtol=1e-2), (
        f"unpermute_reducescatter mismatch in rank {rank}: "
        f"max_diff={( check_fused - check_ref).abs().max().item():.6f}, atol={atol}"
    )
    print(f"[Rank {rank}] Accuracy check PASSED")

    if rank == 0:
        avg_fused = sum(latencies) / len(latencies)
        avg_ref = sum(ref_latencies) / len(ref_latencies)
        print(
            f"[Summary] avg_fused={avg_fused:.3f} ms, "
            f"avg_reference={avg_ref:.3f} ms, speedup={avg_ref / avg_fused:.3f}x"
        )
        print(
            f"[Summary] native_kernel={'yes (_HAS_UNPERMUTE_RS_KERNEL)' if _HAS_UNPERMUTE_RS_KERNEL else 'no (Python fallback)'}"
        )

        if ENABLE_PROJECTION:
            elem_size = expert_output.element_size()

            # Reduce-scatter communication: each rank sends (world_size-1) chunks
            # Each chunk: num_tokens_per_rank * hidden bytes
            rs_bytes = (world_size - 1) * num_tokens_per_rank * hidden_size * elem_size

            # Unpermute memory traffic (per rank):
            #   read:  num_tokens * topk * hidden  (expert_output scatter reads)
            #   write: num_tokens_per_rank * hidden (output)
            read_bytes = num_tokens * topk * hidden_size * elem_size
            write_bytes = num_tokens_per_rank * hidden_size * elem_size
            total_bytes = read_bytes + write_bytes

            proj_rs_ms = project_time_ms(rs_bytes, CROSS_GPU_BW_GBPS)
            proj_compute_ms = project_time_ms(total_bytes, HBM_BW_GBPS)

            print(
                f"[Projection] reduce_scatter_bytes={bytes_to_mb(rs_bytes):.2f} MB "
                f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_rs_ms:.3f} ms"
            )
            print(
                f"[Projection] unpermute_compute (read+write)={bytes_to_mb(total_bytes):.2f} MB "
                f"@{HBM_BW_GBPS:.1f} GB/s -> {proj_compute_ms:.3f} ms"
            )
            print(
                f"[Projection] fused_lower_bound={proj_rs_ms + proj_compute_ms:.3f} ms"
            )
            print(
                f"[Gap Analysis] actual={avg_fused:.3f} ms vs lower_bound={proj_rs_ms + proj_compute_ms:.3f} ms, "
                f"efficiency={( proj_rs_ms + proj_compute_ms) / avg_fused * 100:.1f}%"
            )

    dist.destroy_process_group()


def check_unpermute_allreduce_simple():
    """Standalone smoke test for the simple allreduce baseline (single-rank)."""
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    num_tokens_per_rank = TOKENS_PER_RANK
    num_tokens = num_tokens_per_rank * world_size
    hidden_size = HIDDEN_SIZE
    topk = TOPK

    expert_output, scatter_idx, topk_weights = make_inputs(rank, world_size, device)
    output = torch.zeros(num_tokens_per_rank, hidden_size, device=device, dtype=expert_output.dtype)

    result = unpermute_allreduce_simple(
        expert_output=expert_output,
        scatter_idx=scatter_idx,
        topk_weights=topk_weights,
        output=output,
        group=group,
    )

    assert result.shape == (num_tokens_per_rank, hidden_size), (
        f"Unexpected output shape: {result.shape}"
    )
    assert not result.isnan().any(), "NaN detected in unpermute_allreduce_simple output"
    print(f"[Rank {rank}] unpermute_allreduce_simple smoke test PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    check_unpermute_reducescatter_fusion()
    # check_unpermute_allreduce_simple()
