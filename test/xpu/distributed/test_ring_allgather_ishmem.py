"""Standalone test/benchmark for the ISHMEM ring-allgather op.

Loads libring_allgather_ishmem.so, verifies correctness against
dist.all_gather, and measures latency / bandwidth. Optionally compares against
the full-push allgather_ishmem op (liballgather_permute_ishmem.so).

Run:
    mpirun -np 4 --prepend-rank python test_ring_allgather_ishmem.py

Env:
    TOKENS_PER_RANK (4096), HIDDEN_SIZE (5120), DTYPE (bfloat16)
    LOOP (40), WARMUP (20)
    COMPARE_PUSH=1  also benchmark the full-push allgather_ishmem
"""
import os
import sys
from contextlib import nullcontext

os.environ.setdefault("ISHMEM_IB_ENABLE_IBGDA", "1")
os.environ.setdefault("ISHMEM_IBGDA_DIRECT_DOORBELL", "1")
os.environ.setdefault("ISHMEM_ENABLE_GPU_IPC", "0")
os.environ.setdefault("ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP", "1")
os.environ.setdefault("ISHMEM_SYMMETRIC_SIZE", str(1024 * 1024 * 1024))

import torch
import torch.distributed as dist

TOKENS_PER_RANK = int(os.environ.get("TOKENS_PER_RANK", 1024))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 2048))
LOOP = int(os.environ.get("LOOP", 40))
WARMUP = int(os.environ.get("WARMUP", 20))
COMPARE_PUSH = os.environ.get("COMPARE_PUSH", "0") != "0"
# Enable the PTI-based torch.profiler to capture a chrome trace of the timed
# loops. Set ENABLE_PROFILE=1 to turn on; a per-rank trace is exported.
ENABLE_PROFILE = os.environ.get("ENABLE_PROFILE", "0") != "0"
# Print a progress line every PROGRESS_EVERY iterations of the timed loop so a
# slow run (each ring op can take ~1.8s) does not look like a hang. Set to 0 to
# disable. A progress print forces an xpu.synchronize(), so it also gives a
# rough running latency estimate.
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 5))

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "..", "csrc")


def _load(lib):
    path = os.path.join(_CSRC, lib)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; build it first")
    torch.ops.load_library(path)


def parse_dtype():
    name = os.environ.get("DTYPE", "bfloat16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "half", "float16"):
        return torch.float16
    return torch.float32


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29540")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = rank % torch.xpu.device_count()
    torch.xpu.set_device(dev)
    return rank, world_size, dev


def timed_loop(fn, loop, warmup, progress_rank=None, label=""):
    begin = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    end = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    import time as _time

    wall0 = _time.time()
    for i in range(loop):
        if i >= warmup:
            begin[i].record()
        fn()
        if i >= warmup:
            end[i].record()
        # Periodic progress so a slow run does not look frozen. Forces a
        # synchronize so the printed count reflects real device progress.
        if (
            PROGRESS_EVERY
            and progress_rank is not None
            and (i + 1) % PROGRESS_EVERY == 0
        ):
            torch.xpu.synchronize()
            elapsed = _time.time() - wall0
            print(
                f"[progress rank {progress_rank}] {label} "
                f"{i + 1}/{loop} iters done ({elapsed:.1f}s, "
                f"{elapsed / (i + 1) * 1000:.1f} ms/iter avg)",
                flush=True,
            )
    torch.xpu.synchronize()
    dist.barrier()
    return [begin[i].elapsed_time(end[i]) for i in range(warmup, loop)]


def main():
    rank, world_size, dev = init_distributed()
    device = f"xpu:{dev}"
    dtype = parse_dtype()

    _load("libring_allgather_ishmem.so")
    if COMPARE_PUSH:
        _load("liballgather_permute_ishmem.so")

    torch.manual_seed(1234 + rank)
    shard = torch.randn(TOKENS_PER_RANK, HIDDEN_SIZE, device=device, dtype=dtype)
    gathered = torch.empty(
        TOKENS_PER_RANK * world_size, HIDDEN_SIZE, device=device, dtype=dtype
    )
    print("start to verify correctness \n", flush=True)
    # ---- correctness vs dist.all_gather ----
    torch.ops.symm_mem.ring_allgather_ishmem(shard, gathered, rank, world_size)
    torch.xpu.synchronize()
    print("correctness verify done \n", flush=True)

    ref_list = [torch.empty_like(shard) for _ in range(world_size)]
    dist.all_gather(ref_list, shard)
    ref = torch.cat(ref_list, dim=0)
    ok = torch.equal(gathered, ref)
    max_diff = (gathered.float() - ref.float()).abs().max().item()
    print(f"[ring rank {rank}] correctness exact_match={ok} max_abs_diff={max_diff}", flush=True)

    flag = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0 and flag.item() != 1:
        print("[ring] CORRECTNESS FAILED", flush=True)

    # ---- performance ----
    def run_ring():
        torch.ops.symm_mem.ring_allgather_ishmem(shard, gathered, rank, world_size)
    print("warming up...", flush=True)
    run_ring()
    run_ring()
    torch.xpu.synchronize()
    dist.barrier()
    print("starting timed loop for ring allgather...", flush=True)
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
        ring_lat = timed_loop(run_ring, LOOP, WARMUP, progress_rank=rank, label="ring")

        push_avg = None
        if COMPARE_PUSH:
            print("starting timed loop for push allgather...", flush=True)
            gathered2 = torch.empty_like(gathered)

            def run_push():
                torch.ops.symm_mem.allgather_ishmem(shard, gathered2, rank, world_size)

            run_push()
            run_push()
            torch.xpu.synchronize()
            dist.barrier()
            push_lat = timed_loop(
                run_push, LOOP, WARMUP, progress_rank=rank, label="push"
            )
            push_avg = sum(push_lat) / len(push_lat)

    if ENABLE_PROFILE:
        trace_path = f"./profile_ring_allgather_ishmem_rank{rank}.json"
        prof.export_chrome_trace(trace_path)
        print(f"[ring rank {rank}] profiler trace written to {trace_path}", flush=True)

    ring_avg = sum(ring_lat) / len(ring_lat)
    print(f"ring allgather average latency: {ring_avg:.3f} ms", flush=True)

    if rank == 0:
        elem = shard.element_size()
        # allgather moves (world_size-1) shards worth of data per PE
        bytes_per_pe = (world_size - 1) * TOKENS_PER_RANK * HIDDEN_SIZE * elem
        ring_bw = bytes_per_pe / 1e6 / ring_avg
        print("=" * 68)
        print(
            f"[RING allgather] ws={world_size} tokens/rank={TOKENS_PER_RANK} "
            f"hidden={HIDDEN_SIZE} dtype={dtype}"
        )
        print(
            f"  RING:  avg={ring_avg:.3f} ms  min={min(ring_lat):.3f} "
            f"max={max(ring_lat):.3f}  BW={ring_bw:.2f} GB/s/PE"
        )
        if push_avg is not None:
            push_bw = bytes_per_pe / 1e6 / push_avg
            print(
                f"  PUSH:  avg={push_avg:.3f} ms  BW={push_bw:.2f} GB/s/PE  "
                f"(ring/push={push_avg / ring_avg:.2f}x)"
            )
        print("=" * 68)

    dist.barrier()
    try:
        torch.ops.symm_mem.ring_allgather_ishmem_finalize(torch.empty(0, device=device))
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
