"""
Performance benchmark for ring_reduce_scatter (symmetric-memory single-kernel
ring).

Compares the on-device pipelined ring reduce-scatter against the framework
collective dist.reduce_scatter_tensor, and reports a roofline projection based
on the cross-GPU link bandwidth.

Usage:
    mpirun -n 2 python benchmark_ring_reduce_scatter_dist.py
    mpirun -n 4 python benchmark_ring_reduce_scatter_dist.py
"""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from ring_collectives import (
    _HAS_RING_REDUCE_SCATTER,
    build_ring_reduce_scatter_resources,
    ring_reduce_scatter,
)

# Per-rank output (reduced) block size (elements) and dtype. The full input is
# world_size * CHUNK elements; the reduced output is CHUNK elements.
CHUNK = 2048 * 2048
DTYPE = torch.bfloat16

LOOP = 40
WARMUP = 20
ENABLE_PROFILE = False
ENABLE_PROJECTION = True

PCIE_DISCOUNT = 0.85
CROSS_GPU_BW_GBPS = 31.5 * PCIE_DISCOUNT


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 * 1024)


def project_time_ms(bytes_count, bw_gbps):
    # GB/s is interpreted as 1e9 bytes/s for bandwidth projection.
    return bytes_count / (bw_gbps * 1e9) * 1e3


def init_distributed():
    os.environ.setdefault("RANK", str(os.environ.get("PMI_RANK", 0)))
    os.environ.setdefault("WORLD_SIZE", str(os.environ.get("PMI_SIZE", 1)))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29801")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def benchmark_ring_reduce_scatter():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    if not _HAS_RING_REDUCE_SCATTER:
        if rank == 0:
            print("Native ring_reduce_scatter kernel not built; run csrc/build.py first.")
        dist.destroy_process_group()
        return

    chunk = CHUNK
    elem_size = torch.empty(0, dtype=DTYPE).element_size()

    torch.manual_seed(1234 + rank)
    full = torch.randn(chunk * world_size, dtype=DTYPE, device=device)

    # Pre-allocate symmetric workspace and pointer tensors once, outside the
    # timed loop, so the benchmark measures only the collective itself.
    resources = build_ring_reduce_scatter_resources(full, group=group)

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

    ref_output = torch.empty(chunk, dtype=DTYPE, device=device)

    with prof:
        # Warm up ring reduce-scatter.
        for _ in range(WARMUP):
            out = ring_reduce_scatter(full, group=group, resources=resources)
        torch.xpu.synchronize()
        dist.barrier()

        # Timed ring reduce-scatter.
        for i in range(LOOP):
            if i >= WARMUP:
                begin_events[i].record()
            out = ring_reduce_scatter(full, group=group, resources=resources)
            if i >= WARMUP:
                end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

        # Warm up reference dist.reduce_scatter_tensor.
        for _ in range(WARMUP):
            dist.reduce_scatter_tensor(ref_output, full, op=dist.ReduceOp.SUM, group=group)
        torch.xpu.synchronize()
        dist.barrier()

        # Timed reference dist.reduce_scatter_tensor.
        for i in range(LOOP):
            if i >= WARMUP:
                ref_begin_events[i].record()
            dist.reduce_scatter_tensor(ref_output, full, op=dist.ReduceOp.SUM, group=group)
            if i >= WARMUP:
                ref_end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    ref_latencies = [
        ref_begin_events[i].elapsed_time(ref_end_events[i]) for i in range(WARMUP, LOOP)
    ]

    if ENABLE_PROFILE:
        prof.export_chrome_trace(f"./profile_ring_reduce_scatter_rank{rank}.json")

    # Accuracy check (single fresh pass).
    out = ring_reduce_scatter(full, group=group, resources=resources).clone()
    dist.reduce_scatter_tensor(ref_output, full, op=dist.ReduceOp.SUM, group=group)
    torch.xpu.synchronize()
    # bf16 accumulation order differs; allow a small relative tolerance.
    max_err = (out.float() - ref_output.float()).abs().max().item()
    tol = 1e-2 * ref_output.float().abs().max().clamp_min(1.0).item()
    ok = max_err <= tol

    print(f"[Ring reduce_scatter time in rank {rank}] {latencies} ms")
    print(f"[Reference dist.reduce_scatter_tensor time in rank {rank}] {ref_latencies} ms")

    if rank == 0:
        avg_ring = sum(latencies) / len(latencies)
        avg_ref = sum(ref_latencies) / len(ref_latencies)
        print(f"[Accuracy] ring_reduce_scatter match={ok} max_err={max_err} tol={tol}")
        print(
            f"[Summary] avg_ring={avg_ring:.3f} ms, "
            f"avg_reference={avg_ref:.3f} ms, speedup={avg_ref / avg_ring:.3f}x"
        )

        if ENABLE_PROJECTION:
            # Per-rank moved payload across the ring: (world_size - 1) blocks.
            move_bytes = (world_size - 1) * chunk * elem_size
            proj_ms = project_time_ms(move_bytes, CROSS_GPU_BW_GBPS)
            print(
                f"[Projection] move_bytes={bytes_to_mb(move_bytes):.2f} MB "
                f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_ms:.3f} ms"
            )
            print(
                f"[Gap Analysis] actual={avg_ring:.3f} ms vs projected={proj_ms:.3f} ms, "
                f"ratio={avg_ring / proj_ms:.2f}x, "
                f"efficiency={proj_ms / avg_ring * 100:.1f}%"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    benchmark_ring_reduce_scatter()
