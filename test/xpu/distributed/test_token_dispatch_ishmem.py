"""Standalone test/benchmark for the ISHMEM token-dispatch op.

Each rank holds `TOKENS_PER_RANK` local tokens of shape [tokens, hidden]. Every
token is assigned a random destination rank; the op dispatches each token to its
destination PE's receive buffer using ISHMEM RDMA puts, with all cross-rank
completion signalling done on-device.

Receive layout (per PE): recv_buffer[src * capacity + j] holds the j-th token
that source `src` sent to this PE, and recv_counts[src] is how many it sent.
`capacity == TOKENS_PER_RANK` (worst case: a source sends all its tokens to one
destination).

Run:
    mpirun -np 4 --prepend-rank python test_token_dispatch_ishmem.py

Env:
    TOKENS_PER_RANK (1024), HIDDEN_SIZE (2048), DTYPE (bfloat16)
    LOOP (40), WARMUP (20), SEED (1234)
    ENABLE_PROFILE (1)  report BW from pure kernel time via the PTI profiler
    PROGRESS_EVERY (10) progress print cadence in the timed loop; 0 disables
"""
import os
import sys
import json
from contextlib import nullcontext

os.environ.setdefault("ISHMEM_IB_ENABLE_IBGDA", "1")
os.environ.setdefault("ISHMEM_IBGDA_DIRECT_DOORBELL", "1")
os.environ.setdefault("ISHMEM_ENABLE_GPU_IPC", "0")
os.environ.setdefault("ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP", "1")
os.environ.setdefault("ISHMEM_SYMMETRIC_SIZE", str(2 * 1024 * 1024 * 1024))

import torch
import torch.distributed as dist

TOKENS_PER_RANK = int(os.environ.get("TOKENS_PER_RANK", 8))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 2048))
LOOP = int(os.environ.get("LOOP", 40))
WARMUP = int(os.environ.get("WARMUP", 20))
SEED = int(os.environ.get("SEED", 1234))
# Enable the PTI-based torch.profiler to capture a chrome trace of the timed
# loop so the reported BW is computed from pure dispatch-kernel time and thus
# excludes the host seed/copy-out memcpys the op does around the kernel. Set
# ENABLE_PROFILE=0 to fall back to wall-clock-only timing.
ENABLE_PROFILE = os.environ.get("ENABLE_PROFILE", "1") != "0"
# Print a progress line every PROGRESS_EVERY iterations of the timed loop so a
# slow run does not look like a hang. Set to 0 to disable.
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 10))

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "..", "csrc")
_PROFILED_KERNEL_NAME = "TokenDispatchIshmemKernel"


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
    os.environ.setdefault("MASTER_PORT", "29545")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = rank % torch.xpu.device_count()
    torch.xpu.set_device(dev)
    return rank, world_size, dev


def build_routing(dst_rank, world_size):
    """Precompute (order, send_offsets, send_counts) from per-token dst ranks.

    order  : local token indices sorted by destination (stable, so tokens for a
             given destination keep their original source order).
    send_counts[d]  : number of local tokens destined for d.
    send_offsets[d] : start index into `order` for destination d.
    """
    order = torch.argsort(dst_rank, stable=True).to(torch.int32)
    send_counts = torch.bincount(dst_rank, minlength=world_size).to(torch.int32)
    send_offsets = torch.zeros(world_size, dtype=torch.int32, device=dst_rank.device)
    if world_size > 1:
        send_offsets[1:] = torch.cumsum(send_counts, 0)[:-1].to(torch.int32)
    return order, send_offsets, send_counts


def reference_dispatch(all_tokens, all_dst, rank, world_size, capacity, hidden, dtype, device):
    """Build the expected recv buffer / counts for destination `rank`."""
    expected = torch.zeros(world_size * capacity, hidden, dtype=dtype, device=device)
    expected_counts = torch.zeros(world_size, dtype=torch.int64, device=device)
    for s in range(world_size):
        mask = all_dst[s] == rank
        idxs = mask.nonzero(as_tuple=True)[0]  # original source order
        c = idxs.numel()
        expected_counts[s] = c
        if c > 0:
            expected[s * capacity : s * capacity + c] = all_tokens[s][idxs]
    return expected, expected_counts


def _extract_profiled_kernel_latencies(trace_path, expected_iters):
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    kernel_latencies = []
    for event in events:
        # The kernel functor lives in an unnamed namespace, so the profiler
        # records it as "(anonymous namespace)::TokenDispatchIshmemKernel" (or
        # similarly qualified) rather than the bare class name. Match by
        # substring instead of requiring an exact name.
        if _PROFILED_KERNEL_NAME not in event.get("name", ""):
            continue
        if event.get("ph") != "X":
            continue
        if "dur" not in event:
            continue

        category = event.get("cat", "")
        if category and category not in {"kernel", "gpu_op", "xpu_op"}:
            continue
        kernel_latencies.append(float(event["dur"]) / 1000.0)

    if len(kernel_latencies) < expected_iters:
        raise RuntimeError(
            f"Expected at least {expected_iters} {_PROFILED_KERNEL_NAME} events in "
            f"{trace_path}, found {len(kernel_latencies)}"
        )
    if len(kernel_latencies) > expected_iters:
        kernel_latencies = kernel_latencies[-expected_iters:]
    return kernel_latencies


def _summarize_profiled_kernel(
    rank, world_size, trace_path_fmt, expected_iters, bytes_per_pe=None
):
    # All ranks write their trace to the same directory, so rank 0 can just
    # read every rank's json file straight off disk instead of doing a
    # dist collective to ship the data over.
    dist.barrier()
    if rank != 0:
        return

    gathered = [
        _extract_profiled_kernel_latencies(
            trace_path_fmt.format(rank=r), expected_iters
        )
        for r in range(world_size)
    ]

    for r, rank_latencies in enumerate(gathered):
        rank_avg = sum(rank_latencies) / len(rank_latencies)
        print(
            f"[{_PROFILED_KERNEL_NAME}] rank={r} avg={rank_avg:.3f} ms "
            f"min={min(rank_latencies):.3f} ms max={max(rank_latencies):.3f} ms",
            flush=True,
        )

    # The dispatch completes when the slowest PE's kernel (which internally waits
    # on its peers) finishes, so summarize with the per-iteration MAX across
    # ranks.
    per_iter_max = [
        max(rank_latencies[iter_idx] for rank_latencies in gathered)
        for iter_idx in range(expected_iters)
    ]
    kernel_avg = sum(per_iter_max) / len(per_iter_max)
    print(
        f"[{_PROFILED_KERNEL_NAME}] per-iteration max across ranks/devices: "
        f"{per_iter_max}",
        flush=True,
    )
    print(
        f"[{_PROFILED_KERNEL_NAME}] avg={kernel_avg:.3f} ms "
        f"min={min(per_iter_max):.3f} ms max={max(per_iter_max):.3f} ms",
        flush=True,
    )
    if bytes_per_pe is not None:
        kernel_bw = bytes_per_pe / 1e6 / kernel_avg
        kernel_bw_min = bytes_per_pe / 1e6 / max(per_iter_max)
        kernel_bw_max = bytes_per_pe / 1e6 / min(per_iter_max)
        print(
            f"[{_PROFILED_KERNEL_NAME}] BW avg={kernel_bw:.2f} GB/s/PE "
            f"min={kernel_bw_min:.2f} GB/s/PE max={kernel_bw_max:.2f} GB/s/PE",
            flush=True,
        )


def timed_loop(fn, loop, warmup, progress_rank=None, label=""):
    import time as _time

    begin = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    end = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    dist.barrier()

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
    capacity = TOKENS_PER_RANK

    _load("libtoken_dispatch_ishmem.so")

    torch.manual_seed(SEED + rank)
    tokens = torch.randn(TOKENS_PER_RANK, HIDDEN_SIZE, device=device, dtype=dtype)
    dst_rank = torch.randint(
        0, world_size, (TOKENS_PER_RANK,), device=device, dtype=torch.int32
    )
    order, send_offsets, send_counts = build_routing(dst_rank, world_size)

    recv_buffer = torch.zeros(
        world_size * capacity, HIDDEN_SIZE, device=device, dtype=dtype
    )
    recv_counts = torch.zeros(world_size, device=device, dtype=torch.int64)

    print("start to verify correctness", flush=True)
    torch.ops.symm_mem.token_dispatch_ishmem(
        tokens, order, send_offsets, send_counts,
        recv_buffer, recv_counts, capacity, rank, world_size,
    )
    torch.xpu.synchronize()

    # ---- correctness ----
    all_tokens = [torch.empty_like(tokens) for _ in range(world_size)]
    dist.all_gather(all_tokens, tokens)
    all_dst = [torch.empty_like(dst_rank) for _ in range(world_size)]
    dist.all_gather(all_dst, dst_rank)

    expected, expected_counts = reference_dispatch(
        all_tokens, all_dst, rank, world_size, capacity, HIDDEN_SIZE, dtype, device
    )

    counts_ok = torch.equal(recv_counts, expected_counts)
    assert counts_ok, (
        f"[rank {rank}] recv_counts mismatch: got {recv_counts.tolist()} "
        f"expected {expected_counts.tolist()}"
    )
    for s in range(world_size):
        c = int(expected_counts[s].item())
        if c == 0:
            continue
        got = recv_buffer[s * capacity : s * capacity + c]
        ref = expected[s * capacity : s * capacity + c]
        assert torch.equal(got, ref), (
            f"[rank {rank}] token data mismatch for source {s} "
            f"(max_abs_diff={(got.float() - ref.float()).abs().max().item()})"
        )
    print(
        f"[rank {rank}] correctness OK counts={recv_counts.tolist()} "
        f"total_received={int(recv_counts.sum().item())}",
        flush=True,
    )

    # ---- performance ----
    def run():
        torch.ops.symm_mem.token_dispatch_ishmem(
            tokens, order, send_offsets, send_counts,
            recv_buffer, recv_counts, capacity, rank, world_size,
        )

    run()
    run()
    torch.xpu.synchronize()
    dist.barrier()

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
        lat = timed_loop(run, LOOP, WARMUP, progress_rank=rank, label="dispatch")

    avg = sum(lat) / len(lat)
    # Bytes each PE pushes out per dispatch: every local token is put exactly
    # once (self-dispatch included as a local put).
    elem = tokens.element_size()
    bytes_per_pe = TOKENS_PER_RANK * HIDDEN_SIZE * elem

    if ENABLE_PROFILE:
        trace_path = f"./profile_token_dispatch_ishmem_rank{rank}.json"
        prof.export_chrome_trace(trace_path)
        print(f"[rank {rank}] profiler trace written to {trace_path}", flush=True)
        # Kernel-time BW excludes the host seed/copy-out memcpys, so it reflects
        # the actual dispatch rather than the end-to-end op.
        _summarize_profiled_kernel(
            rank,
            world_size,
            "./profile_token_dispatch_ishmem_rank{rank}.json",
            len(lat),
            bytes_per_pe=bytes_per_pe,
        )

    if rank == 0:
        bw = bytes_per_pe / 1e6 / avg
        print("=" * 68)
        print(
            f"[TOKEN dispatch] ws={world_size} tokens/rank={TOKENS_PER_RANK} "
            f"hidden={HIDDEN_SIZE} dtype={dtype} capacity={capacity}"
        )
        print(
            f"  end2end: avg={avg:.3f} ms  min={min(lat):.3f}  max={max(lat):.3f}  "
            f"BW={bw:.2f} GB/s/PE (sent, incl. host copies)"
        )
        print("=" * 68)

    dist.barrier()
    try:
        torch.ops.symm_mem.token_dispatch_ishmem_finalize(
            torch.empty(0, device=device)
        )
    except Exception as e:
        print(f"[rank {rank}] finalize raised: {e!r}", flush=True)
    dist.destroy_process_group()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
