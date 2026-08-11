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
from contextlib import nullcontext

os.environ.setdefault("ISHMEM_IB_ENABLE_IBGDA", "1")
os.environ.setdefault("ISHMEM_IBGDA_DIRECT_DOORBELL", "1")
os.environ.setdefault("ISHMEM_ENABLE_GPU_IPC", "0")
os.environ.setdefault("ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP", "1")
os.environ.setdefault("ISHMEM_SYMMETRIC_SIZE", str(2 * 1024 * 1024 * 1024))

import torch
import torch.distributed as dist

TOKENS_PER_RANK = int(os.environ.get("TOKENS_PER_RANK", 1024))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 4096))
LOOP = int(os.environ.get("LOOP", 40))
WARMUP = int(os.environ.get("WARMUP", 20))
SEED = int(os.environ.get("SEED", 1234))
ENABLE_PROFILE = os.environ.get("ENABLE_PROFILE", "1") != "0"
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", 10))

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

    # ---- correctness ----
    all_tokens = [torch.empty_like(tokens) for _ in range(world_size)]
    dist.all_gather(all_tokens, tokens)
    all_dst = [torch.empty_like(dst_rank) for _ in range(world_size)]
    dist.all_gather(all_dst, dst_rank)

    expected, expected_counts = reference_dispatch(
        all_tokens, all_dst, rank, world_size, capacity, HIDDEN_SIZE, dtype, device
    )

    assert torch.equal(recv_counts, expected_counts), (
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

    avg = sum(lat) / len(lat)
    elem = tokens.element_size()
    bytes_per_pe = TOKENS_PER_RANK * HIDDEN_SIZE * elem

    if ENABLE_PROFILE:
        trace_path = f"./profile_token_dispatch_ishmem_hier_rank{rank}.json"
        prof.export_chrome_trace(trace_path)
        print(f"[rank {rank}] profiler trace written to {trace_path}", flush=True)

    if rank == 0:
        bw = bytes_per_pe / 1e6 / avg
        print("=" * 68)
        print(
            f"[TOKEN dispatch HIER] ws={world_size} pcie_domain={pcie_domain} "
            f"tokens/rank={TOKENS_PER_RANK} hidden={HIDDEN_SIZE} dtype={dtype} "
            f"capacity={capacity}"
        )
        print(
            f"  end2end: avg={avg:.3f} ms  min={min(lat):.3f}  max={max(lat):.3f}  "
            f"BW={bw:.2f} GB/s/PE (sent, incl. host copies)"
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
