"""
Performance benchmark for ring_allgather (symmetric-memory single-kernel ring).

Compares the on-device pipelined ring allgather against the framework
collective dist.all_gather, and reports a roofline projection based on the
cross-GPU link bandwidth.

Usage:
    mpirun -n 2 python benchmark_ring_allgather_dist.py
    mpirun -n 4 python benchmark_ring_allgather_dist.py
"""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

import env

from ring_collectives import (
    _HAS_RING_ALLGATHER,
    build_ring_allgather_resources,
    ring_allgather,
)

# Per-rank shard size (elements) and dtype.
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
    env.setup_distributed_env(
        master_addr="127.0.0.1", master_port="29801", overwrite=False
    )
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def benchmark_ring_allgather():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    if not _HAS_RING_ALLGATHER:
        if rank == 0:
            print("Native ring_allgather kernel not built; run csrc/build.py first.")
        dist.destroy_process_group()
        return

    chunk = CHUNK
    elem_size = torch.empty(0, dtype=DTYPE).element_size()

    torch.manual_seed(1234 + rank)
    shard = torch.randn(chunk, dtype=DTYPE, device=device)

    # Pre-allocate symmetric workspace and pointer tensors once, outside the
    # timed loop, so the benchmark measures only the collective itself.
    resources = build_ring_allgather_resources(shard, group=group)

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

    gathered = [torch.empty(chunk, dtype=DTYPE, device=device) for _ in range(world_size)]

    with prof:
        # Warm up ring allgather.
        for _ in range(WARMUP):
            out = ring_allgather(shard, group=group, resources=resources)
        torch.xpu.synchronize()
        dist.barrier()

        # Timed ring allgather.
        for i in range(LOOP):
            if i >= WARMUP:
                begin_events[i].record()
            out = ring_allgather(shard, group=group, resources=resources)
            if i >= WARMUP:
                end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

        # Warm up reference dist.all_gather.
        for _ in range(WARMUP):
            dist.all_gather(gathered, shard, group=group)
        torch.xpu.synchronize()
        dist.barrier()

        # Timed reference dist.all_gather.
        for i in range(LOOP):
            if i >= WARMUP:
                ref_begin_events[i].record()
            dist.all_gather(gathered, shard, group=group)
            if i >= WARMUP:
                ref_end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    ref_latencies = [
        ref_begin_events[i].elapsed_time(ref_end_events[i]) for i in range(WARMUP, LOOP)
    ]

    if ENABLE_PROFILE:
        prof.export_chrome_trace(f"./profile_ring_allgather_rank{rank}.json")

    # Accuracy check (single fresh pass).
    out = ring_allgather(shard, group=group, resources=resources).clone()
    dist.all_gather(gathered, shard, group=group)
    expected = torch.cat(gathered, dim=0)
    torch.xpu.synchronize()
    ok = torch.equal(out, expected)

    print(f"[Ring allgather time in rank {rank}] {latencies} ms")
    print(f"[Reference dist.all_gather time in rank {rank}] {ref_latencies} ms")

    if rank == 0:
        avg_ring = sum(latencies) / len(latencies)
        avg_ref = sum(ref_latencies) / len(ref_latencies)
        print(f"[Accuracy] ring_allgather match={ok}")
        print(
            f"[Summary] avg_ring={avg_ring:.3f} ms, "
            f"avg_reference={avg_ref:.3f} ms, speedup={avg_ref / avg_ring:.3f}x"
        )

        if ENABLE_PROJECTION:
            # Per-rank received payload: (world_size - 1) shards.
            recv_bytes = (world_size - 1) * chunk * elem_size
            proj_ms = project_time_ms(recv_bytes, CROSS_GPU_BW_GBPS)
            print(
                f"[Projection] recv_bytes={bytes_to_mb(recv_bytes):.2f} MB "
                f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_ms:.3f} ms"
            )
            print(
                f"[Gap Analysis] actual={avg_ring:.3f} ms vs projected={proj_ms:.3f} ms, "
                f"ratio={avg_ring / proj_ms:.2f}x, "
                f"efficiency={proj_ms / avg_ring * 100:.1f}%"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    benchmark_ring_allgather()
