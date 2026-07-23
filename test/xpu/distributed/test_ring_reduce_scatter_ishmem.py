"""Standalone test/benchmark for the ISHMEM ring reduce-scatter op.

Loads libring_reduce_scatter_ishmem.so, verifies correctness against
dist.reduce_scatter_tensor, and measures latency / bandwidth.

Run:
    mpirun -np 4 --prepend-rank python test_ring_reduce_scatter_ishmem.py

Env:
    CHUNK (2048)          per-rank output element count (the reduced block)
    DTYPE (bfloat16)
    LOOP (40), WARMUP (20)
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

CHUNK = int(os.environ.get("CHUNK", 2048))
LOOP = int(os.environ.get("LOOP", 40))
WARMUP = int(os.environ.get("WARMUP", 20))
# Enable the PTI-based torch.profiler to capture a chrome trace of the timed
# loops. Set ENABLE_PROFILE=1 to turn on; a per-rank trace is exported.
ENABLE_PROFILE = os.environ.get("ENABLE_PROFILE", "0") != "0"
# Print a progress line every PROGRESS_EVERY iterations of the timed loop so a
# slow run does not look like a hang. Set to 0 to disable.
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
    os.environ.setdefault("MASTER_PORT", "29541")
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

    _load("libring_reduce_scatter_ishmem.so")

    # Each rank holds the full input of world_size blocks, block b is
    # contributed towards the reduced output of rank b.
    torch.manual_seed(1234 + rank)
    inp = torch.randn(CHUNK * world_size, device=device, dtype=dtype)
    output = torch.empty(CHUNK, device=device, dtype=dtype)

    print("start to verify correctness \n", flush=True)
    # ---- correctness vs dist.reduce_scatter_tensor ----
    torch.ops.symm_mem.ring_reduce_scatter_ishmem(inp, output, rank, world_size)
    torch.xpu.synchronize()
    print("correctness verify done \n", flush=True)

    ref = torch.empty(CHUNK, device=device, dtype=dtype)
    dist.reduce_scatter_tensor(ref, inp.contiguous(), op=dist.ReduceOp.SUM)

    # bf16/fp16 accumulate in float on device; allow a small tolerance.
    if dtype == torch.float32:
        ok = torch.equal(output, ref)
    else:
        ok = torch.allclose(output.float(), ref.float(), rtol=1e-2, atol=1e-2)
    max_diff = (output.float() - ref.float()).abs().max().item()
    print(
        f"[ring rank {rank}] correctness match={ok} max_abs_diff={max_diff}",
        flush=True,
    )

    flag = torch.tensor([1 if ok else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0 and flag.item() != 1:
        print("[ring] CORRECTNESS FAILED", flush=True)

    # ---- performance ----
    def run_ring():
        torch.ops.symm_mem.ring_reduce_scatter_ishmem(inp, output, rank, world_size)

    print("warming up...", flush=True)
    run_ring()
    run_ring()
    torch.xpu.synchronize()
    dist.barrier()
    print("starting timed loop for ring reduce-scatter...", flush=True)
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

    if ENABLE_PROFILE:
        trace_path = f"./profile_ring_reduce_scatter_ishmem_rank{rank}.json"
        prof.export_chrome_trace(trace_path)
        print(f"[ring rank {rank}] profiler trace written to {trace_path}", flush=True)

    ring_avg = sum(ring_lat) / len(ring_lat)
    print(f"ring reduce-scatter average latency: {ring_avg:.3f} ms", flush=True)

    if rank == 0:
        elem = inp.element_size()
        # reduce-scatter moves (world_size-1) blocks worth of data per PE
        bytes_per_pe = (world_size - 1) * CHUNK * elem
        ring_bw = bytes_per_pe / 1e6 / ring_avg
        print("=" * 68)
        print(
            f"[RING reduce-scatter] ws={world_size} chunk={CHUNK} dtype={dtype}"
        )
        print(
            f"  RING:  avg={ring_avg:.3f} ms  min={min(ring_lat):.3f} "
            f"max={max(ring_lat):.3f}  BW={ring_bw:.2f} GB/s/PE"
        )
        print("=" * 68)

    dist.barrier()
    try:
        torch.ops.symm_mem.ring_reduce_scatter_ishmem_finalize(
            torch.empty(0, device=device)
        )
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
