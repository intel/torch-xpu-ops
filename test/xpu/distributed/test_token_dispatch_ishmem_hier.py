"""Standalone test/benchmark for the HIERARCHICAL ISHMEM token-dispatch op.

Two-PCIE-domain topology: PCIE_DOMAIN == P cards per domain, exactly two domains
(world_size == 2 * P). Dispatch is done in two on-device kernels:

  Kernel 1 (cross-domain RDMA, mirror push): rank r sends all its cross-domain
  tokens to its mirror partner (r + P) % world_size.
  Kernel 2 (intra-domain): each rank forwards the staged tokens plus its own
  same-domain tokens to their final destination inside the domain.

The output is byte-identical to the flat token_dispatch_ishmem op:
recv_buffer[src * capacity + j] holds the j-th token that source `src` sent to
this PE (in the source's original order), and recv_counts[src] is that count.

Run:
    mpirun -np 4 --prepend-rank python test_token_dispatch_ishmem_hier.py

Env:
    TOKENS_PER_RANK (1024), HIDDEN_SIZE (2048), DTYPE (bfloat16)
    PCIE_DOMAIN (world_size // 2)
    LOOP (40), WARMUP (20), SEED (1234)
    ENABLE_PROFILE (1)  export a chrome trace of the timed loop
    PROGRESS_EVERY (10) progress print cadence; 0 disables
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

TOKENS_PER_RANK = int(os.environ.get("TOKENS_PER_RANK", 4096))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 7168))
LOOP = int(os.environ.get("LOOP", 40))
WARMUP = int(os.environ.get("WARMUP", 20))
SEED = int(os.environ.get("SEED", 1234))
ENABLE_PROFILE = os.environ.get("ENABLE_PROFILE", "1") != "0"
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 10))

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "..", "csrc")
# The dispatch runs two kernels per iteration; both live in an unnamed namespace
# so the profiler qualifies them (e.g. "(anonymous namespace)::..."). Match by
# substring.
_PROFILED_KERNEL_NAMES = (
    "TokenDispatchIshmemHierK1",
)


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
    os.environ.setdefault("MASTER_PORT", "29546")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dev = rank % torch.xpu.device_count()
    torch.xpu.set_device(dev)
    return rank, world_size, dev


def build_hier_routing(dst_rank, rank, world_size, pcie_domain, capacity):
    """Precompute the hierarchical routing arrays for the local tokens.

    Returns (cross_order, cross_slot, cross_destpos, local_order, local_slot,
    local_destpos, recv_counts_in), where:
      *_order   : local token indices (int32)
      *_slot    : the absolute final slot src*capacity+j the token must land in
      *_destpos : the destination position within its domain (0..P-1)
      recv_counts_in[s] : number of tokens original source s sends to this rank.

    cross_* are the tokens whose destination is in the OTHER domain (kernel 1
    then kernel 2); local_* are the tokens whose destination is in THIS domain
    (kernel 2 only, includes self).
    """
    P = pcie_domain
    device = dst_rank.device
    S = dst_rank.numel()
    dst = dst_rank.to(torch.int64)

    # j = running index of each token within its (src->dst) group, original order.
    j = torch.empty(S, dtype=torch.int64, device=device)
    for d in range(world_size):
        idx = (dst == d).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            j[idx] = torch.arange(idx.numel(), device=device)

    final_slot = (rank * capacity + j).to(torch.int32)
    destpos = (dst % P).to(torch.int32)
    my_domain = rank // P
    cross_mask = (dst // P) != my_domain
    local_mask = ~cross_mask
    tok_idx = torch.arange(S, dtype=torch.int32, device=device)

    cross_order = tok_idx[cross_mask].contiguous()
    cross_slot = final_slot[cross_mask].contiguous()
    cross_destpos = destpos[cross_mask].contiguous()
    local_order = tok_idx[local_mask].contiguous()
    local_slot = final_slot[local_mask].contiguous()
    local_destpos = destpos[local_mask].contiguous()

    # recv_counts_in via all-gather of the per-rank send-count vectors.
    send_counts = torch.bincount(dst, minlength=world_size).to(torch.int64)
    gathered = [torch.empty_like(send_counts) for _ in range(world_size)]
    dist.all_gather(gathered, send_counts)
    matrix = torch.stack(gathered)  # [src, dst]
    recv_counts_in = matrix[:, rank].contiguous().to(torch.int64)

    return (
        cross_order,
        cross_slot,
        cross_destpos,
        local_order,
        local_slot,
        local_destpos,
        recv_counts_in,
    )


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


def _extract_kernel_latencies(trace_path, kernel_name, expected_iters):
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    kernel_latencies = []
    for event in trace.get("traceEvents", []):
        if kernel_name not in event.get("name", ""):
            continue
        if event.get("ph") != "X" or "dur" not in event:
            continue
        category = event.get("cat", "")
        if category and category not in {"kernel", "gpu_op", "xpu_op"}:
            continue
        kernel_latencies.append(float(event["dur"]) / 1000.0)

    if len(kernel_latencies) < expected_iters:
        raise RuntimeError(
            f"Expected at least {expected_iters} {kernel_name} events in "
            f"{trace_path}, found {len(kernel_latencies)}"
        )
    return kernel_latencies[-expected_iters:]


def _extract_profiled_dispatch_latencies(trace_path, expected_iters):
    """Per-iteration end-to-end kernel time = K1 + K2 (both on-device)."""
    per_kernel = [
        _extract_kernel_latencies(trace_path, name, expected_iters)
        for name in _PROFILED_KERNEL_NAMES
    ]
    return [sum(k[i] for k in per_kernel) for i in range(expected_iters)]


def _summarize_profiled_kernel(
    rank, world_size, trace_path_fmt, expected_iters, bytes_per_rank=None
):
    # All ranks write their trace to the same directory, so rank 0 reads every
    # rank's json straight off disk.
    dist.barrier()
    if rank != 0:
        return

    gathered = [
        _extract_profiled_dispatch_latencies(
            trace_path_fmt.format(rank=r), expected_iters
        )
        for r in range(world_size)
    ]

    for r, rank_latencies in enumerate(gathered):
        rank_avg = sum(rank_latencies) / len(rank_latencies)
        rank_bw = None
        if bytes_per_rank is not None:
            rank_bw = bytes_per_rank[r] / 1e6 / rank_avg
        print(
            f"[dispatch_hier kernel] rank={r} avg={rank_avg:.3f} ms "
            f"min={min(rank_latencies):.3f} ms max={max(rank_latencies):.3f} ms"
            + (
                f" BW={rank_bw:.2f} GB/s/PE (real cross bytes)"
                if rank_bw is not None
                else ""
            ),
            flush=True,
        )

    # The dispatch completes when the slowest PE's kernels finish, so summarize
    # with the per-iteration MAX across ranks.
    per_iter_max = [
        max(rank_latencies[iter_idx] for rank_latencies in gathered)
        for iter_idx in range(expected_iters)
    ]
    kernel_avg = sum(per_iter_max) / len(per_iter_max)
    print(
        f"[dispatch_hier kernel] per-iteration max across ranks/devices: "
        f"{per_iter_max}",
        flush=True,
    )
    print(
        f"[dispatch_hier kernel] avg={kernel_avg:.3f} ms "
        f"min={min(per_iter_max):.3f} ms max={max(per_iter_max):.3f} ms",
        flush=True,
    )
    if bytes_per_rank is not None:
        rank_bws = [
            bytes_per_rank[r] / 1e6 / (sum(gathered[r]) / len(gathered[r]))
            for r in range(world_size)
        ]
        print(
            f"[dispatch_hier kernel] BW per-rank GB/s "
            f"{[round(v, 2) for v in rank_bws]} (real cross bytes)",
            flush=True,
        )


def timed_loop(fn, loop, warmup, progress_rank=None, label=""):
    import time as _time

    dist.barrier()
    torch.xpu.synchronize()

    # Warmup iterations: not timed, no progress prints.
    for _ in range(warmup):
        fn()

    torch.xpu.synchronize()
    # Pure host (wall-clock) timing of the timed region: no per-iter sycl
    # events, since inserting an enable_timing event around each launch
    # perturbs the in-order queue's kernel scheduling/PTI-reported durations.
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


def main():
    rank, world_size, dev = init_distributed()
    device = f"xpu:{dev}"
    dtype = parse_dtype()
    capacity = TOKENS_PER_RANK
    pcie_domain = int(os.environ.get("PCIE_DOMAIN", world_size // 2))
    assert world_size == 2 * pcie_domain, (
        f"hierarchical dispatch requires exactly two domains: world_size="
        f"{world_size} pcie_domain={pcie_domain}"
    )

    _load("libtoken_dispatch_ishmem_hier.so")

    torch.manual_seed(SEED + rank)
    tokens = torch.randn(TOKENS_PER_RANK, HIDDEN_SIZE, device=device, dtype=dtype)
    dst_rank = torch.randint(
        0, world_size, (TOKENS_PER_RANK,), device=device, dtype=torch.int32
    )
    (
        cross_order,
        cross_slot,
        cross_destpos,
        local_order,
        local_slot,
        local_destpos,
        recv_counts_in,
    ) = build_hier_routing(dst_rank, rank, world_size, pcie_domain, capacity)

    print(
        f"[rank {rank}] num_cross={cross_order.numel()} "
        f"num_local={local_order.numel()} (total={TOKENS_PER_RANK})",
        flush=True,
    )

    recv_buffer = torch.zeros(
        world_size * capacity, HIDDEN_SIZE, device=device, dtype=dtype
    )
    recv_counts = torch.zeros(world_size, device=device, dtype=torch.int64)

    def run():
        torch.ops.symm_mem.token_dispatch_ishmem_hier(
            tokens,
            cross_order,
            cross_slot,
            cross_destpos,
            local_order,
            local_slot,
            local_destpos,
            recv_buffer,
            recv_counts,
            recv_counts_in,
            capacity,
            rank,
            world_size,
            pcie_domain,
        )

    print("start to verify correctness", flush=True)
    recv_buffer.zero_()
    run()
    torch.xpu.synchronize()

    # ---- correctness (mirror dispatch only) ----
    # This op currently runs ONLY the cross-domain mirror exchange (kernel 1):
    # rank r receives, into the front of recv_buffer, exactly the cross-domain
    # tokens its mirror partner (r + P) % world_size sent, in that partner's
    # cross_order (stable original order). Verify that staged payload.
    all_tokens = [torch.empty_like(tokens) for _ in range(world_size)]
    dist.all_gather(all_tokens, tokens)
    all_dst = [torch.empty_like(dst_rank) for _ in range(world_size)]
    dist.all_gather(all_dst, dst_rank)

    P = pcie_domain
    mirror = (rank + P) % world_size
    mirror_dst = all_dst[mirror]
    # Mirror's cross tokens are those bound for a domain other than the mirror's
    # own, i.e. THIS rank's domain (only two domains exist).
    mirror_cross_mask = (mirror_dst // P) != (mirror // P)
    cross_positions = mirror_cross_mask.nonzero(as_tuple=True)[0]  # original order
    num_staged = int(cross_positions.numel())
    expected_staged = all_tokens[mirror][cross_positions]

    got = recv_buffer[:num_staged]
    assert torch.equal(got, expected_staged), (
        f"[rank {rank}] staged mirror token mismatch (num_staged={num_staged}, "
        f"max_abs_diff={(got.float() - expected_staged.float()).abs().max().item()})"
    )
    print(
        f"[rank {rank}] mirror-dispatch correctness OK "
        f"num_staged={num_staged} (from mirror rank {mirror})",
        flush=True,
    )

    # ---- performance ----
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
        lat = timed_loop(run, LOOP, WARMUP, progress_rank=rank, label="dispatch_hier")

    elem = tokens.element_size()
    cross_tokens_local = int(cross_order.numel())
    cross_tokens_t = torch.tensor([cross_tokens_local], dtype=torch.int64, device=device)
    gathered_cross_tokens = [torch.empty_like(cross_tokens_t) for _ in range(world_size)]
    dist.all_gather(gathered_cross_tokens, cross_tokens_t)
    bytes_per_rank = [
        int(t.item()) * HIDDEN_SIZE * elem for t in gathered_cross_tokens
    ]
    avg = sum(lat) / len(lat)
    avg_t = torch.tensor([avg], dtype=torch.float64, device=device)
    gathered_avg = [torch.empty_like(avg_t) for _ in range(world_size)]
    dist.all_gather(gathered_avg, avg_t)
    avg_per_rank = [float(t.item()) for t in gathered_avg]

    if ENABLE_PROFILE:
        trace_path = f"./profile_token_dispatch_ishmem_hier_rank{rank}.json"
        prof.export_chrome_trace(trace_path)
        print(f"[rank {rank}] profiler trace written to {trace_path}", flush=True)
        # Kernel-time BW parsed from the trace excludes the host seed/copy-out
        # memcpys, so it reflects the actual dispatch rather than the end-to-end op.
        _summarize_profiled_kernel(
            rank,
            world_size,
            "./profile_token_dispatch_ishmem_hier_rank{rank}.json",
            len(lat),
            bytes_per_rank=bytes_per_rank,
        )

    if rank == 0:
        rank_bws = [
            bytes_per_rank[r] / 1e6 / avg_per_rank[r]
            for r in range(world_size)
        ]
        print("=" * 68)
        print(
            f"[TOKEN dispatch HIER] ws={world_size} pcie_domain={pcie_domain} "
            f"tokens/rank={TOKENS_PER_RANK} hidden={HIDDEN_SIZE} dtype={dtype} "
            f"capacity={capacity}"
        )
        print(
            f"  end2end: avg={avg:.3f} ms  min={min(lat):.3f}  max={max(lat):.3f}  "
            f"BW(rank0)={rank_bws[0]:.2f} GB/s/PE (real cross bytes, incl. host copies)"
        )
        print(
            f"  end2end BW per-rank GB/s {[round(v, 2) for v in rank_bws]} "
            f"(real cross bytes, each rank uses its own avg latency)"
        )
        print("=" * 68)

    dist.barrier()
    try:
        torch.ops.symm_mem.token_dispatch_ishmem_hier_finalize(
            torch.empty(0, device=device)
        )
    except Exception as e:
        print(f"[rank {rank}] finalize raised: {e!r}", flush=True)
    dist.destroy_process_group()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
