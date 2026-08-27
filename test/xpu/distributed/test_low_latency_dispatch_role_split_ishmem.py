"""Standalone correctness + performance test for the ISHMEM low-latency,
role-split MoE dispatch op (`symm_mem.low_latency_dispatch_role_split_ishmem`).

This is a simplified, single-kernel reproducer of DeepSymm's
`LowLatencyDispatchRoleSplitKernelBK` (moe_ep/internode_ll.cpp): work-groups
are split into two roles launched together in ONE kernel --
  - "expert" WGs (one per global expert id) scan this rank's local tokens and
    RDMA-push (ISHMEM) every token routed to their expert to the owning
    rank's receive buffer, then publish a completion flag/count;
  - "receiver" WGs (one per local expert owned by this rank) wait on that
    flag from every source rank and gather the arrived tokens into
    `packed_recv_x` / `packed_recv_src_info`, DeepEP-style.

Each rank holds `TOKENS_PER_RANK` local tokens of shape [tokens, hidden].
Each token independently selects `TOPK` experts out of `NUM_EXPERTS`, which
are uniformly sharded across `WORLD_SIZE` ranks/devices.

Run:
    mpirun -np 4 --prepend-rank python \
        test_low_latency_dispatch_role_split_ishmem.py

Env:
    TOKENS_PER_RANK (128), HIDDEN_SIZE (2048), DTYPE (bfloat16)
    TOPK (8), NUM_EXPERTS (32), CAPACITY_MULT (2)
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

TOKENS_PER_RANK = int(os.environ.get("TOKENS_PER_RANK", 128))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 2048))
LOOP = int(os.environ.get("LOOP", 40))
WARMUP = int(os.environ.get("WARMUP", 20))
SEED = int(os.environ.get("SEED", 1234))
TOPK = int(os.environ.get("TOPK", 8))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 32))
# Per (local_expert, src_rank) slot capacity, expressed as a multiple of the
# expected average arrivals (TOKENS_PER_RANK * TOPK / NUM_EXPERTS), so random
# routing doesn't overflow slots and silently drop tokens in this test.
CAPACITY_MULT = int(os.environ.get("CAPACITY_MULT", 4))
ENABLE_PROFILE = os.environ.get("ENABLE_PROFILE", "1") != "0"
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 10))

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "..", "csrc")
_PROFILED_KERNEL_NAME = "LowLatencyDispatchRoleSplitIshmemKernel"


def _load(lib):
    path = os.path.join(_CSRC, lib)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; build it first (see build.py)")
    torch.ops.load_library(path)


def parse_dtype():
    name = os.environ.get("DTYPE", "bfloat16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError("low_latency_dispatch_role_split_ishmem only supports bfloat16")


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29546")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = rank % torch.xpu.device_count()
    torch.xpu.set_device(dev)
    return rank, world_size, dev


def reference_dispatch(
    all_tokens, all_topk, rank, world_size, num_local_experts, capacity, hidden, dtype, device
):
    """Build the expected packed_recv_x/src_info/count/layout_range for `rank`.

    Mirrors the kernel's per-(local_expert, src_rank) slot assignment: within
    a given source rank, a local expert's arrivals keep the source rank's
    original token order (the kernel claims slots via a monotonically
    increasing atomic counter as it scans tokens 0..num_tokens-1).
    """
    expected_x = torch.zeros(
        num_local_experts * world_size * capacity, hidden, dtype=dtype, device=device
    )
    expected_src = torch.full(
        (num_local_experts * world_size * capacity,), -1, dtype=torch.int32, device=device
    )
    expected_count = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    expected_layout = torch.zeros(
        num_local_experts * world_size, dtype=torch.int64, device=device
    )

    for local_expert in range(num_local_experts):
        expert = rank * num_local_experts + local_expert
        begin = 0
        for src_rank in range(world_size):
            topk = all_topk[src_rank]  # [tokens, TOPK]
            hits = (topk == expert).any(dim=1).nonzero(as_tuple=True)[0]
            count = min(hits.numel(), capacity)
            hits = hits[:count]
            flag_idx = local_expert * world_size + src_rank
            expected_layout[flag_idx] = (int(begin) & 0xFFFFFFFF) << 32 | (count & 0xFFFFFFFF)
            dst_base = local_expert * world_size * capacity + begin
            if count > 0:
                expected_x[dst_base : dst_base + count] = all_tokens[src_rank][hits]
                expected_src[dst_base : dst_base + count] = hits.to(torch.int32)
            expected_count[local_expert] += count
            begin += count
    return expected_x, expected_src, expected_count, expected_layout


def timed_loop(fn, loop, warmup, progress_rank=None, label=""):
    import time as _time

    dist.barrier()
    torch.xpu.synchronize()
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    timed_iters = loop - warmup
    wall0 = _time.time()
    for i in range(timed_iters):
        fn()
        if (
            PROGRESS_EVERY
            and progress_rank is not None
            and (i + 1) % PROGRESS_EVERY == 0
        ):
            torch.xpu.synchronize()
            elapsed = _time.time() - wall0
            print(
                f"[progress rank {progress_rank}] {label} "
                f"{i + 1}/{timed_iters} iters done ({elapsed:.1f}s, "
                f"{elapsed / (i + 1) * 1000:.1f} ms/iter avg)",
                flush=True,
            )
    torch.xpu.synchronize()
    timed_wall = (_time.time() - wall0) * 1000.0
    dist.barrier()
    per_iter = timed_wall / timed_iters
    return [per_iter for _ in range(timed_iters)]


def _extract_profiled_kernel_latencies(trace_path, expected_iters):
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)
    events = trace.get("traceEvents", [])
    kernel_latencies = []
    for event in events:
        if _PROFILED_KERNEL_NAME not in event.get("name", ""):
            continue
        if event.get("ph") != "X" or "dur" not in event:
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


def _summarize_profiled_kernel(rank, world_size, trace_path_fmt, expected_iters, bytes_per_pe=None):
    dist.barrier()
    if rank != 0:
        return
    gathered = [
        _extract_profiled_kernel_latencies(trace_path_fmt.format(rank=r), expected_iters)
        for r in range(world_size)
    ]
    for r, rank_latencies in enumerate(gathered):
        rank_avg = sum(rank_latencies) / len(rank_latencies)
        print(
            f"[{_PROFILED_KERNEL_NAME}] rank={r} avg={rank_avg:.3f} ms "
            f"min={min(rank_latencies):.3f} ms max={max(rank_latencies):.3f} ms",
            flush=True,
        )
    per_iter_max = [
        max(rank_latencies[iter_idx] for rank_latencies in gathered)
        for iter_idx in range(expected_iters)
    ]
    kernel_avg = sum(per_iter_max) / len(per_iter_max)
    print(
        f"[{_PROFILED_KERNEL_NAME}] per-iteration max across ranks/devices: {per_iter_max}",
        flush=True,
    )
    print(
        f"[{_PROFILED_KERNEL_NAME}] avg={kernel_avg:.3f} ms "
        f"min={min(per_iter_max):.3f} ms max={max(per_iter_max):.3f} ms",
        flush=True,
    )
    if bytes_per_pe is not None:
        kernel_bw = bytes_per_pe / 1e6 / kernel_avg
        print(f"[{_PROFILED_KERNEL_NAME}] BW avg={kernel_bw:.2f} GB/s/PE", flush=True)


def main():
    rank, world_size, dev = init_distributed()
    device = f"xpu:{dev}"
    dtype = parse_dtype()

    if TOPK <= 0 or TOPK > NUM_EXPERTS:
        raise ValueError(f"invalid TOPK={TOPK} for NUM_EXPERTS={NUM_EXPERTS}")
    if NUM_EXPERTS % world_size != 0:
        raise ValueError(
            f"NUM_EXPERTS must be divisible by WORLD_SIZE, got {NUM_EXPERTS}/{world_size}"
        )

    num_local_experts = NUM_EXPERTS // world_size
    avg_per_slot = max(1, (TOKENS_PER_RANK * TOPK) // NUM_EXPERTS)
    capacity = max(TOKENS_PER_RANK, avg_per_slot * CAPACITY_MULT)

    _load("liblow_latency_dispatch_role_split_ishmem.so")

    torch.manual_seed(SEED + rank)
    x = torch.randn(TOKENS_PER_RANK, HIDDEN_SIZE, device=device, dtype=dtype)
    probs = torch.ones(TOKENS_PER_RANK, NUM_EXPERTS, device=device, dtype=torch.float32)
    topk_idx = torch.multinomial(probs, TOPK, replacement=False).to(torch.int32).contiguous()
    if os.environ.get("LL_PRINT_TOPK", "0") == "1":
        print(f"[rank {rank}] topk_idx={topk_idx.cpu().tolist()}", flush=True)

    packed_recv_x = torch.zeros(
        num_local_experts * world_size * capacity, HIDDEN_SIZE, device=device, dtype=dtype
    )
    packed_recv_src_info = torch.full(
        (num_local_experts * world_size * capacity,), -1, device=device, dtype=torch.int32
    )
    packed_recv_count = torch.zeros(num_local_experts, device=device, dtype=torch.int32)
    packed_recv_layout_range = torch.zeros(
        num_local_experts * world_size, device=device, dtype=torch.int64
    )

    if rank == 0:
        print(
            f"[config] ws={world_size} num_experts={NUM_EXPERTS} topk={TOPK} "
            f"experts/rank={num_local_experts} tokens/rank={TOKENS_PER_RANK} "
            f"hidden={HIDDEN_SIZE} capacity={capacity}",
            flush=True,
        )

    print("start to verify correctness", flush=True)
    torch.ops.symm_mem.low_latency_dispatch_role_split_ishmem(
        x, topk_idx, packed_recv_x, packed_recv_src_info, packed_recv_count,
        packed_recv_layout_range, capacity, NUM_EXPERTS, rank, world_size,
    )
    torch.xpu.synchronize()

    # ---- correctness ----
    all_tokens = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(all_tokens, x)
    all_topk = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(all_topk, topk_idx)

    exp_x, exp_src, exp_count, exp_layout = reference_dispatch(
        all_tokens, all_topk, rank, world_size, num_local_experts, capacity,
        HIDDEN_SIZE, dtype, device,
    )

    assert torch.equal(packed_recv_count, exp_count), (
        f"[rank {rank}] packed_recv_count mismatch: got {packed_recv_count.tolist()} "
        f"expected {exp_count.tolist()}"
    )
    assert torch.equal(packed_recv_layout_range, exp_layout), (
        f"[rank {rank}] packed_recv_layout_range mismatch: got "
        f"{packed_recv_layout_range.tolist()} expected {exp_layout.tolist()}"
    )
    assert torch.equal(packed_recv_x, exp_x), (
        f"[rank {rank}] packed_recv_x mismatch "
        f"(max_abs_diff={(packed_recv_x.float() - exp_x.float()).abs().max().item()})"
    )
    assert torch.equal(packed_recv_src_info, exp_src), (
        f"[rank {rank}] packed_recv_src_info mismatch"
    )
    print(
        f"[rank {rank}] correctness OK counts={packed_recv_count.tolist()} "
        f"total_received={int(packed_recv_count.sum().item())}",
        flush=True,
    )

    # ---- performance ----
    def run():
        torch.ops.symm_mem.low_latency_dispatch_role_split_ishmem(
            x, topk_idx, packed_recv_x, packed_recv_src_info, packed_recv_count,
            packed_recv_layout_range, capacity, NUM_EXPERTS, rank, world_size,
        )

    print(f"[rank {rank}] warmup run 1 begin", flush=True)
    run()
    print(f"[rank {rank}] warmup run 1 end", flush=True)
    print(f"[rank {rank}] warmup run 2 begin", flush=True)
    run()
    print(f"[rank {rank}] warmup run 2 end", flush=True)
    print(f"[rank {rank}] post-warmup synchronize begin", flush=True)
    torch.xpu.synchronize()
    print(f"[rank {rank}] post-warmup synchronize end", flush=True)
    print(f"[rank {rank}] post-warmup barrier begin", flush=True)
    dist.barrier()
    print(f"[rank {rank}] post-warmup barrier end", flush=True)

    if ENABLE_PROFILE:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()

    print(f"[rank {rank}] timed loop begin", flush=True)
    with prof:
        lat = timed_loop(run, LOOP, WARMUP, progress_rank=rank, label="ll_dispatch_role_split")
    print(f"[rank {rank}] timed loop end", flush=True)

    avg = sum(lat) / len(lat)
    elem = x.element_size()
    bytes_per_pe = TOKENS_PER_RANK * TOPK * HIDDEN_SIZE * elem

    if ENABLE_PROFILE:
        trace_path = f"./profile_ll_dispatch_role_split_ishmem_rank{rank}.json"
        prof.export_chrome_trace(trace_path)
        print(f"[rank {rank}] profiler trace written to {trace_path}", flush=True)
        _summarize_profiled_kernel(
            rank,
            world_size,
            "./profile_ll_dispatch_role_split_ishmem_rank{rank}.json",
            len(lat),
            bytes_per_pe=bytes_per_pe,
        )

    if rank == 0:
        bw = bytes_per_pe / 1e6 / avg
        print("=" * 68)
        print(
            f"[LL dispatch role-split] ws={world_size} tokens/rank={TOKENS_PER_RANK} "
            f"topk={TOPK} experts={NUM_EXPERTS} hidden={HIDDEN_SIZE} "
            f"dtype={dtype} capacity={capacity}"
        )
        print(
            f"  end2end: avg={avg:.3f} ms  min={min(lat):.3f}  max={max(lat):.3f}  "
            f"BW={bw:.2f} GB/s/PE (sent, incl. host copies)"
        )
        print("=" * 68)

    print(f"[rank {rank}] final barrier begin", flush=True)
    dist.barrier()
    print(f"[rank {rank}] final barrier end", flush=True)
    print(f"[rank {rank}] ishmem finalize begin", flush=True)
    try:
        torch.ops.symm_mem.low_latency_dispatch_role_split_ishmem_finalize(
            torch.empty(0, device=device)
        )
    except Exception as e:
        print(f"[rank {rank}] finalize raised: {e!r}", flush=True)
    print(f"[rank {rank}] ishmem finalize end", flush=True)
    if os.environ.get("LL_SKIP_DIST_DESTROY", "0") == "1":
        print(f"[rank {rank}] skipping destroy process group", flush=True)
        sys.stdout.flush()
        os._exit(0)
    print(f"[rank {rank}] destroy process group begin", flush=True)
    dist.destroy_process_group()
    print(f"[rank {rank}] destroy process group end", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
