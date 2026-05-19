"""
Accuracy + performance check for allgather_local_permute_fusion (distributed style)

Usage:
    mpirun -n 2 python test_allgather_local_permute_fusion_dist.py
"""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from allgather_local_permute_fusion import (
    allgather_local_permute_fusion,
    allgather_with_symm_mem,
    build_allgather_rank_buffers_ptr,
    compute_scatter_idx,
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


def run_reference_allgather_local_permute(
    hidden_shard,
    output_tensor,
    scatter_idx,
    group,
    world_size,
    num_tokens_per_rank,
    topk,
    gathered_ref,
):
    dist.all_gather(gathered_ref, hidden_shard, group=group)
    all_hidden_ref = torch.stack(gathered_ref, dim=0)  # [world_size, tokens_per_rank, hidden_size]
    for src_rank in range(world_size):
        token_offset = src_rank * num_tokens_per_rank
        torch.ops.symm_mem.local_permute_copy_(
            all_hidden_ref[src_rank],
            scatter_idx,
            token_offset,
            output_tensor,
        )

def init_distributed():
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29513'
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size

def check_allgather_local_permute_fusion():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    num_tokens_per_rank = TOKENS_PER_RANK
    hidden_size = HIDDEN_SIZE
    topk = TOPK
    num_tokens = num_tokens_per_rank * world_size
    backend_stream = torch.xpu.Stream()

    # Each rank: unique hidden_shard
    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(num_tokens_per_rank, hidden_size, device=device, dtype=torch.bfloat16)

    # All ranks: same topk_idx
    torch.manual_seed(42)
    topk_idx = torch.randint(0, world_size, (num_tokens, topk), device=device, dtype=torch.int64)
    scatter_idx, _ = compute_scatter_idx(topk_idx)

    # Precompute rank_buffers_ptr (also allocates workspace)
    rank_buffers = build_allgather_rank_buffers_ptr(hidden_shard, group=group)

    # Reference path uses only current stream: allgather first, then local permute.
    gathered_ref = [torch.empty_like(hidden_shard) for _ in range(world_size)]

    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    ref_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    ref_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    backend_stream = torch.xpu.Stream()

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
        # Warm up fused path
        for _ in range(WARMUP):
            remap_hidden_states = torch.empty((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)
            output_fused = allgather_local_permute_fusion(
                hidden_shard,
                topk_idx,
                scatter_idx,
                remap_hidden_states=remap_hidden_states,
                group=group,
                backend_stream=backend_stream,
                rank_buffers_ptr=rank_buffers,
            )
        torch.xpu.synchronize()
        dist.barrier()

        remap_hidden_states = torch.empty((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)
        # Timed fused path
        for i in range(LOOP):
            if i >= WARMUP:
                begin_events[i].record()
            output_fused = allgather_local_permute_fusion(
                hidden_shard,
                topk_idx,
                scatter_idx,
                remap_hidden_states=remap_hidden_states,
                group=group,
                backend_stream=backend_stream,
                rank_buffers_ptr=rank_buffers,
            )
            if i >= WARMUP:
                end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

        # Warm up reference path: allgather + local permute (current stream only)
        ref_output = torch.empty((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)
        for _ in range(WARMUP):
            run_reference_allgather_local_permute(
                hidden_shard,
                ref_output,
                scatter_idx,
                group,
                world_size,
                num_tokens_per_rank,
                topk,
                gathered_ref,
            )
        torch.xpu.synchronize()
        dist.barrier()

        # Timed reference path
        for i in range(LOOP):
            if i >= WARMUP:
                ref_begin_events[i].record()
            run_reference_allgather_local_permute(
                hidden_shard,
                ref_output,
                scatter_idx,
                group,
                world_size,
                num_tokens_per_rank,
                topk,
                gathered_ref,
            )
            if i >= WARMUP:
                ref_end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

        # --- Isolated local permute timing (warm cache) ---
        # Pre-gather data so we only time the permute part
        dist.all_gather(gathered_ref, hidden_shard, group=group)
        all_hidden_pregathered = torch.stack(gathered_ref, dim=0)
        torch.xpu.synchronize()
        dist.barrier()

        perm_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
        perm_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

        perm_output = torch.empty((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)
        # Warm up local permute only
        for _ in range(WARMUP):
            for src_rank in range(world_size):
                token_offset = src_rank * num_tokens_per_rank
                torch.ops.symm_mem.local_permute_copy_(
                    all_hidden_pregathered[src_rank],
                    scatter_idx,
                    token_offset,
                    perm_output,
                )
        torch.xpu.synchronize()

        # Timed local permute only (warm cache — same input every iteration)
        for i in range(LOOP):
            if i >= WARMUP:
                perm_begin_events[i].record()
            for src_rank in range(world_size):
                token_offset = src_rank * num_tokens_per_rank
                torch.ops.symm_mem.local_permute_copy_(
                    all_hidden_pregathered[src_rank],
                    scatter_idx,
                    token_offset,
                    perm_output,
                )
            if i >= WARMUP:
                perm_end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

        # --- Isolated local permute timing (cold cache) ---
        # Use rotating input buffers to defeat cache: total > L2 capacity
        NUM_ROTATE_BUFS = 8  # 8 × 32 MB = 256 MB input cycling
        rotate_inputs = []
        for r in range(NUM_ROTATE_BUFS):
            gathered_tmp = [torch.empty_like(hidden_shard) for _ in range(world_size)]
            dist.all_gather(gathered_tmp, hidden_shard, group=group)
            rotate_inputs.append(torch.stack(gathered_tmp, dim=0))
        torch.xpu.synchronize()
        dist.barrier()

        cold_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
        cold_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

        cold_output = torch.empty((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)
        # Warm up cold path
        for w in range(WARMUP):
            inp = rotate_inputs[w % NUM_ROTATE_BUFS]
            for src_rank in range(world_size):
                token_offset = src_rank * num_tokens_per_rank
                torch.ops.symm_mem.local_permute_copy_(
                    inp[src_rank],
                    scatter_idx,
                    token_offset,
                    cold_output,
                )
        torch.xpu.synchronize()

        # Timed cold-cache permute: rotate through different input buffers
        for i in range(LOOP):
            inp = rotate_inputs[i % NUM_ROTATE_BUFS]
            if i >= WARMUP:
                cold_begin_events[i].record()
            for src_rank in range(world_size):
                token_offset = src_rank * num_tokens_per_rank
                torch.ops.symm_mem.local_permute_copy_(
                    inp[src_rank],
                    scatter_idx,
                    token_offset,
                    cold_output,
                )
            if i >= WARMUP:
                cold_end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

        # --- Fused single-launch permute timing ---
        fused_begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
        fused_end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
        fused_output = torch.empty((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)

        has_fused = hasattr(torch.ops.symm_mem, "local_permute_copy_fused_")
        if has_fused:
            # Warm up fused kernel
            for _ in range(WARMUP):
                torch.ops.symm_mem.local_permute_copy_fused_(
                    all_hidden_pregathered, scatter_idx, fused_output)
            torch.xpu.synchronize()

            # Timed fused permute
            for i in range(LOOP):
                if i >= WARMUP:
                    fused_begin_events[i].record()
                torch.ops.symm_mem.local_permute_copy_fused_(
                    all_hidden_pregathered, scatter_idx, fused_output)
                if i >= WARMUP:
                    fused_end_events[i].record()
            torch.xpu.synchronize()
            dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    ref_latencies = [ref_begin_events[i].elapsed_time(ref_end_events[i]) for i in range(WARMUP, LOOP)]
    perm_latencies = [perm_begin_events[i].elapsed_time(perm_end_events[i]) for i in range(WARMUP, LOOP)]
    cold_latencies = [cold_begin_events[i].elapsed_time(cold_end_events[i]) for i in range(WARMUP, LOOP)]
    fused_perm_latencies = [fused_begin_events[i].elapsed_time(fused_end_events[i]) for i in range(WARMUP, LOOP)] if has_fused else []

    if ENABLE_PROFILE:
        prof.export_chrome_trace(f"./profile_allgather_local_permute_fusion_rank{rank}.json")

    print(f"[Fusion time in rank {rank}] {latencies} ms")
    print(f"[Reference allgather+permute time in rank {rank}] {ref_latencies} ms")
    print(f"[Local permute only (warm) time in rank {rank}] {perm_latencies} ms")
    print(f"[Local permute only (cold) time in rank {rank}] {cold_latencies} ms")
    if fused_perm_latencies:
        print(f"[Fused permute (single launch) time in rank {rank}] {fused_perm_latencies} ms")

    # Accuracy check: run one fresh pass and compare against reference
    remap_check = torch.zeros((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)
    allgather_local_permute_fusion(
        hidden_shard, topk_idx, scatter_idx, remap_hidden_states=remap_check, group=group,
        backend_stream=backend_stream, rank_buffers_ptr=rank_buffers,
    )
    ref = torch.zeros_like(remap_check)
    run_reference_allgather_local_permute(
        hidden_shard,
        ref,
        scatter_idx,
        group,
        world_size,
        num_tokens_per_rank,
        topk,
        gathered_ref,
    )
    assert torch.equal(remap_check, ref), f"allgather_local_permute_fusion mismatch in rank {rank}"

    if rank == 0:
        avg_fused = sum(latencies) / len(latencies)
        avg_ref = sum(ref_latencies) / len(ref_latencies)
        avg_perm = sum(perm_latencies) / len(perm_latencies)
        avg_cold = sum(cold_latencies) / len(cold_latencies)
        avg_fused_perm = sum(fused_perm_latencies) / len(fused_perm_latencies) if fused_perm_latencies else 0
        print(
            f"[Summary] avg_fused={avg_fused:.3f} ms, "
            f"avg_reference={avg_ref:.3f} ms, speedup={avg_ref / avg_fused:.3f}x"
        )
        print(f"[Summary] avg_local_permute_warm={avg_perm:.3f} ms, avg_local_permute_cold={avg_cold:.3f} ms")
        if avg_fused_perm > 0:
            print(f"[Summary] avg_fused_single_launch={avg_fused_perm:.3f} ms")

        if ENABLE_PROJECTION:
            elem_size = hidden_shard.element_size()

            # Per-rank communication payload received from peers for allgather.
            allgather_bytes = (world_size - 1) * num_tokens_per_rank * hidden_size * elem_size

            # Local permute memory traffic breakdown:
            read_bytes = num_tokens * hidden_size * elem_size
            write_bytes = num_tokens * topk * hidden_size * elem_size
            total_bytes = read_bytes + write_bytes  # cold: full HBM traffic

            proj_allgather_ms = project_time_ms(allgather_bytes, CROSS_GPU_BW_GBPS)
            proj_cold_ms = project_time_ms(total_bytes, HBM_BW_GBPS)
            proj_write_only_ms = project_time_ms(write_bytes, HBM_BW_GBPS)

            print(
                f"[Projection] allgather_bytes={bytes_to_mb(allgather_bytes):.2f} MB "
                f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_allgather_ms:.3f} ms"
            )
            print(
                f"[Projection] local_permute cold (read+write)={bytes_to_mb(total_bytes):.2f} MB "
                f"@{HBM_BW_GBPS:.1f} GB/s -> {proj_cold_ms:.3f} ms"
            )
            print(
                f"[Projection] local_permute warm (write-only)={bytes_to_mb(write_bytes):.2f} MB "
                f"@{HBM_BW_GBPS:.1f} GB/s -> {proj_write_only_ms:.3f} ms"
            )
            print(
                f"[Projection] fused_lower_bound={proj_allgather_ms + proj_cold_ms:.3f} ms"
            )

            best_perm = avg_fused_perm if avg_fused_perm > 0 else avg_perm
            best_label = "fused_single_launch" if avg_fused_perm > 0 else "4-launch"
            print(
                f"\n[Gap Analysis - Cold] actual={avg_cold:.3f} ms vs projected={proj_cold_ms:.3f} ms, "
                f"ratio={avg_cold / proj_cold_ms:.2f}x, "
                f"efficiency={proj_cold_ms / avg_cold * 100:.1f}%"
            )
            print(
                f"[Gap Analysis - Warm] actual={avg_perm:.3f} ms vs projected={proj_write_only_ms:.3f} ms, "
                f"ratio={avg_perm / proj_write_only_ms:.2f}x, "
                f"efficiency={proj_write_only_ms / avg_perm * 100:.1f}%"
            )
            if avg_fused_perm > 0:
                print(
                    f"[Gap Analysis - Fused] actual={avg_fused_perm:.3f} ms vs projected={proj_cold_ms:.3f} ms, "
                    f"ratio={avg_fused_perm / proj_cold_ms:.2f}x, "
                    f"efficiency={proj_cold_ms / avg_fused_perm * 100:.1f}%"
                )

    dist.destroy_process_group()

def check_allgather_with_symm_mem():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    numel_per_rank = TOKENS_PER_RANK * HIDDEN_SIZE
    total_numel = numel_per_rank * world_size

    # Each rank: unique input
    torch.manual_seed(1234 + rank)
    input_shard = torch.randn(numel_per_rank, device=device, dtype=torch.bfloat16)

    print(f"[Allgather input size per rank] {input_shard.numel() * input_shard.element_size() / 1024 / 1024:.2f} MB")

    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

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
        # Warm up
        for _ in range(WARMUP):
            output = torch.empty(total_numel, device=device, dtype=input_shard.dtype)
            allgather_with_symm_mem(
                input_shard,
                output_tensor=output,
                group=group,
            )
        torch.xpu.synchronize()
        dist.barrier()

        output = torch.empty(total_numel, device=device, dtype=input_shard.dtype)
        # Timed path
        for i in range(LOOP):
            if i >= WARMUP:
                begin_events[i].record()
            allgather_with_symm_mem(
                input_shard,
                output_tensor=output,
                group=group,
            )
            if i >= WARMUP:
                end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]

    if ENABLE_PROFILE:
        prof.export_chrome_trace(f"./profile_allgather_with_symm_mem_rank{rank}.json")

    print(f"[Allgather time in rank {rank}] {latencies} ms")

    # Accuracy check: gather all input_shard from all ranks and compare
    gathered = [torch.empty_like(input_shard) for _ in range(world_size)]
    dist.all_gather(gathered, input_shard, group=group)
    ref = torch.cat(gathered, dim=0)
    assert torch.allclose(output, ref, atol=1e-3), f"Allgather mismatch in rank {rank}"

    if rank == 0:
        avg = sum(latencies) / len(latencies)
        print(f"[Summary] allgather avg={avg:.3f} ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    check_allgather_local_permute_fusion()
    #check_allgather_with_symm_mem()
