"""Correctness UT for the ISHMEM `internode_dispatch_rdma_sender` op.

This op is a standalone port of DeepSymm's `InternodeDispatchRDMASenderKernel`
(intel-sandbox/DeepSymm: csrc/modules/moe_ep/internode.cpp) -- the first
("RDMA sender") stage of the legacy two-level internode MoE dispatch. Ranks
are grouped into nodes of `NUM_MAX_NVL_PEERS` (2) PEs each; each token carries
a per-GLOBAL-rank `is_token_in_rank` bit, which this kernel ORs into a small
per-destination-NODE bitmask and stages+sends the token payload (hidden data +
top-k indices/weights + source metadata) to every node whose bitmask is
non-zero: a local copy for the sender's own node, an ISHMEM RDMA put for every
other node.

This test only exercises that one kernel (no forward/receive stage), so the
observable output is each node's own RECEIVE region: recv_x/recv_topk_idx/
recv_topk_weights/recv_src_rdma_rank/recv_src_nvl_bits (all indexed by source
node + slot) and recv_counts (how many tokens each source node sent).

Run (4 ranks == 2 nodes x 2 NVL peers/node):
    mpirun -np 4 --prepend-rank python test_internode_dispatch_rdma_sender.py

Env:
    NUM_TOKENS (16), HIDDEN_SIZE (1024), TOPK (8), EXPERTS_PER_RANK (8),
    NUM_MAX_TOKENS_PER_RANK (32), SEED (1234)
    DTYPE (float32)
    ENABLE_PROFILE (1)  use the PTI-based torch.profiler to capture a chrome
                        trace of the timed loop and report BW from pure
                        InternodeDispatchRdmaSenderKernel time (falls back to
                        wall-clock-only timing if disabled)
    LOOP (30), WARMUP (10)  perf-loop iteration counts (after correctness)
"""
import json
import os
import sys
import time
from contextlib import nullcontext

os.environ.setdefault("ISHMEM_IB_ENABLE_IBGDA", "1")
os.environ.setdefault("ISHMEM_IBGDA_DIRECT_DOORBELL", "1")
os.environ.setdefault("ISHMEM_ENABLE_GPU_IPC", "0")
os.environ.setdefault("ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP", "1")
os.environ.setdefault("ISHMEM_SYMMETRIC_SIZE", str(512 * 1024 * 1024))

import torch
import torch.distributed as dist

NUM_MAX_NVL_PEERS = 2

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", 16))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 1024))
TOPK = int(os.environ.get("TOPK", 8))
EXPERTS_PER_RANK = int(os.environ.get("EXPERTS_PER_RANK", 8))
num_max_tokens_per_rank = int(os.environ.get("NUM_MAX_TOKENS_PER_RANK", 32))
SEED = int(os.environ.get("SEED", 1234))
ENABLE_PROFILE = os.environ.get("ENABLE_PROFILE", "1") != "0"
LOOP = int(os.environ.get("LOOP", 30))
WARMUP = int(os.environ.get("WARMUP", 10))

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "..", "csrc")
_PROFILED_KERNEL_NAME = "InternodeDispatchRdmaSenderKernel"


def _load(lib):
    path = os.path.join(_CSRC, lib)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; build it first (python build_ishmem.py)")
    torch.ops.load_library(path)


def parse_dtype():
    name = os.environ.get("DTYPE", "float32").lower()
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


def _extract_profiled_kernel_latencies(trace_path, expected_iters):
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    kernel_latencies = []
    for event in events:
        # The kernel functor lives in an unnamed namespace, so the profiler
        # records it as "(anonymous namespace)::InternodeDispatchRdmaSenderKernel"
        # (or similarly qualified) rather than the bare class name. Match by
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


def main():
    rank, world_size, dev = init_distributed()
    assert world_size % NUM_MAX_NVL_PEERS == 0, (
        f"world_size must be a multiple of NUM_MAX_NVL_PEERS={NUM_MAX_NVL_PEERS}, "
        f"got {world_size}"
    )
    num_rdma_ranks = world_size // NUM_MAX_NVL_PEERS
    my_rdma = rank // NUM_MAX_NVL_PEERS
    device = f"xpu:{dev}"
    dtype = parse_dtype()

    _load("libinternode_dispatch_rdma_sender.so")

    torch.manual_seed(SEED + rank)
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE, device=device, dtype=dtype)
    # MoE-style routing: each token picks TOPK distinct experts uniformly, and
    # is routed to the ranks that own those experts (rank = expert // EXPERTS_PER_RANK).
    num_experts = world_size * EXPERTS_PER_RANK
    assert TOPK <= num_experts, f"TOPK={TOPK} must be <= num_experts={num_experts}"
    topk_idx = (
        torch.rand(NUM_TOKENS, num_experts, device=device)
        .argsort(dim=1)[:, :TOPK]
        .to(torch.int64)
    )
    topk_weights = torch.rand(NUM_TOKENS, TOPK, device=device, dtype=torch.float32)
    dest_rank = topk_idx // EXPERTS_PER_RANK
    is_token_in_rank = torch.zeros(NUM_TOKENS, world_size, dtype=torch.bool, device=device)
    is_token_in_rank.scatter_(1, dest_rank, True)

    recv_x = torch.zeros(num_rdma_ranks, num_max_tokens_per_rank, HIDDEN_SIZE, device=device, dtype=dtype)
    recv_topk_idx = torch.zeros(num_rdma_ranks, num_max_tokens_per_rank, TOPK, device=device, dtype=torch.int64)
    recv_topk_weights = torch.zeros(num_rdma_ranks, num_max_tokens_per_rank, TOPK, device=device, dtype=torch.float32)
    recv_src_rdma_rank = torch.zeros(num_rdma_ranks * num_max_tokens_per_rank, device=device, dtype=torch.int32)
    recv_src_nvl_bits = torch.zeros(num_rdma_ranks * num_max_tokens_per_rank, device=device, dtype=torch.int32)
    recv_counts = torch.zeros(num_rdma_ranks, device=device, dtype=torch.int64)

    torch.ops.symm_mem.internode_dispatch_rdma_sender(
        x,
        None,
        topk_idx,
        topk_weights,
        is_token_in_rank,
        recv_x,
        None,
        recv_topk_idx,
        recv_topk_weights,
        recv_src_rdma_rank,
        recv_src_nvl_bits,
        recv_counts,
        rank,
        world_size,
        num_max_tokens_per_rank,
        0,
    )
    torch.xpu.synchronize()

    # ---- build ground truth via all_gather ----
    all_x = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(all_x, x)
    all_topk_idx = [torch.empty_like(topk_idx) for _ in range(world_size)]
    dist.all_gather(all_topk_idx, topk_idx)
    all_topk_weights = [torch.empty_like(topk_weights) for _ in range(world_size)]
    dist.all_gather(all_topk_weights, topk_weights)
    all_in_rank = [torch.empty_like(is_token_in_rank) for _ in range(world_size)]
    dist.all_gather(all_in_rank, is_token_in_rank)

    # NOTE on lane pairing: the ported kernel's RDMA put target is
    # `dst_pe = rd * NUM_MAX_NVL_PEERS + my_nvl`, where `my_nvl` is the
    # SENDING rank's own NVL lane (not the destination bit) -- so receiver
    # (dst_node, L) only ever receives from sender (src_node, L): the SAME
    # lane index on both ends. Fanning a token out to a *different* NVL lane
    # on the destination node is the job of the (out-of-scope-for-this-port)
    # forward/receive stage, which redistributes using the SourceMeta bits
    # this kernel stashes. So the expected receive set for this rank is only
    # what the single same-lane sender staged for our node, not the whole
    # source node's aggregate.
    expected_counts = torch.zeros(num_rdma_ranks, dtype=torch.int64)
    expected_rows = {}
    for src_node in range(num_rdma_ranks):
        src_rank = src_node * NUM_MAX_NVL_PEERS + (rank % NUM_MAX_NVL_PEERS)
        in_rank_row = all_in_rank[src_rank]
        bits = torch.zeros(in_rank_row.size(0), dtype=torch.int32, device=in_rank_row.device)
        for j in range(NUM_MAX_NVL_PEERS):
            dst_rank = my_rdma * NUM_MAX_NVL_PEERS + j
            bits |= (in_rank_row[:, dst_rank].to(torch.int32)) << j
        mask = bits != 0
        idxs = mask.nonzero(as_tuple=True)[0]
        rows_x = all_x[src_rank][idxs].cpu()
        rows_topk_idx = all_topk_idx[src_rank][idxs].cpu()
        rows_topk_w = all_topk_weights[src_rank][idxs].cpu()
        rows_bits = bits[idxs].cpu()
        cnt = rows_x.size(0)
        expected_counts[src_node] = min(cnt, num_max_tokens_per_rank)
        expected_rows[src_node] = (rows_x, rows_topk_idx, rows_topk_w, rows_bits)

    counts_ok = torch.equal(recv_counts.cpu(), expected_counts)
    assert counts_ok, (
        f"[rank {rank}] recv_counts mismatch: got {recv_counts.tolist()} "
        f"expected {expected_counts.tolist()}"
    )

    for src_node in range(num_rdma_ranks):
        c = int(expected_counts[src_node].item())
        if c == 0:
            continue
        got_x = recv_x[src_node, :c].cpu()
        got_topk_idx = recv_topk_idx[src_node, :c].cpu()
        got_topk_w = recv_topk_weights[src_node, :c].cpu()
        got_bits = recv_src_nvl_bits[src_node * num_max_tokens_per_rank : src_node * num_max_tokens_per_rank + c].cpu()
        got_src_rank = recv_src_rdma_rank[src_node * num_max_tokens_per_rank : src_node * num_max_tokens_per_rank + c].cpu()
        assert torch.equal(got_src_rank, torch.full((c,), src_node, dtype=torch.int32)), (
            f"[rank {rank}] src_rdma_rank mismatch for source node {src_node}: "
            f"{got_src_rank.tolist()}"
        )

        exp_x, exp_topk_idx, exp_topk_w, exp_bits = expected_rows[src_node]
        # Canonicalize row order (the atomic-add slot race across a source
        # node's NUM_MAX_NVL_PEERS ranks is nondeterministic): sort both sides
        # by (bits, hidden-row sum, topk-idx sum) as a stable-enough key.
        def sort_key(xt, ti, tw, bt):
            key = (
                bt.to(torch.float64) * 1e12
                + xt.double().sum(dim=1) * 1e6
                + ti.double().sum(dim=1)
            )
            order = torch.argsort(key)
            return xt[order], ti[order], tw[order], bt[order]

        got_x, got_topk_idx, got_topk_w, got_bits = sort_key(got_x, got_topk_idx, got_topk_w, got_bits)
        exp_x, exp_topk_idx, exp_topk_w, exp_bits = sort_key(exp_x, exp_topk_idx, exp_topk_w, exp_bits)

        assert torch.equal(got_bits, exp_bits), (
            f"[rank {rank}] nvl bits mismatch for source node {src_node}: "
            f"got={got_bits.tolist()} expected={exp_bits.tolist()}"
        )
        assert torch.equal(got_topk_idx, exp_topk_idx), (
            f"[rank {rank}] topk_idx mismatch for source node {src_node}"
        )
        assert torch.allclose(got_topk_w.float(), exp_topk_w.float(), atol=1e-6), (
            f"[rank {rank}] topk_weights mismatch for source node {src_node}"
        )
        assert torch.allclose(got_x.float(), exp_x.float(), atol=1e-3), (
            f"[rank {rank}] token payload mismatch for source node {src_node} "
            f"(max_abs_diff={(got_x.float() - exp_x.float()).abs().max().item()})"
        )

    print(
        f"[rank {rank}] internode_dispatch_rdma_sender correctness OK "
        f"counts={recv_counts.tolist()}",
        flush=True,
    )

    # ---- performance (PTI profiler) ----
    # Re-run the op in a loop, optionally under torch.profiler (PTI XPU
    # backend), to report the InternodeDispatchRdmaSenderKernel's own time.
    # No dist.barrier() is used for pacing: the op's internal
    # ishmem_barrier_all() already keeps every rank lock-step per call, which
    # is enough to pipeline the timed loop without an extra (and, on this
    # NIC-only ISHMEM setup, occasionally flaky) collective python barrier.
    def run():
        torch.ops.symm_mem.internode_dispatch_rdma_sender(
            x,
            None,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            recv_x,
            None,
            recv_topk_idx,
            recv_topk_weights,
            recv_src_rdma_rank,
            recv_src_nvl_bits,
            recv_counts,
            rank,
            world_size,
            num_max_tokens_per_rank,
            0,
        )

    for _ in range(WARMUP):
        run()
    torch.xpu.synchronize()

    timed_iters = max(LOOP - WARMUP, 1)
    prof = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
        if ENABLE_PROFILE
        else nullcontext()
    )

    wall0 = time.time()
    with prof:
        for _ in range(timed_iters):
            run()
        torch.xpu.synchronize()
    wall_ms = (time.time() - wall0) * 1000.0 / timed_iters

    # Bytes this rank actually sends over RDMA per call: only tokens routed to
    # a REMOTE node count (the sender's own node is a local device copy, not
    # RDMA). Per destination node we send min(tokens_routed_there, cap) token
    # payloads; sum those over all remote nodes.
    elem_size = x.element_size()
    src_meta_bytes = 8  # int32 src_rdma_rank + int32 bits
    bytes_per_token = HIDDEN_SIZE * elem_size + src_meta_bytes + TOPK * 8 + TOPK * 4
    bytes_per_token = (bytes_per_token + 15) // 16 * 16
    # A token hits node rd if any of that node's NUM_MAX_NVL_PEERS lanes is set;
    # cap at the per-node slot capacity and drop this rank's own (local) node.
    node_any = is_token_in_rank.view(NUM_TOKENS, num_rdma_ranks, NUM_MAX_NVL_PEERS).any(dim=2)
    tokens_per_node = node_any.sum(dim=0).clamp(max=num_max_tokens_per_rank)
    tokens_per_node[my_rdma] = 0
    rdma_tokens = int(tokens_per_node.sum().item())
    bytes_per_iter = rdma_tokens * bytes_per_token

    if ENABLE_PROFILE:
        trace_path = f"./profile_internode_dispatch_rdma_sender_rank{rank}.json"
        prof.export_chrome_trace(trace_path)
        try:
            kernel_latencies = _extract_profiled_kernel_latencies(trace_path, timed_iters)
            kernel_avg = sum(kernel_latencies) / len(kernel_latencies)
            kernel_bw = bytes_per_iter / 1e6 / kernel_avg
            print(
                f"[rank {rank}] [{_PROFILED_KERNEL_NAME}] avg={kernel_avg:.3f} ms "
                f"min={min(kernel_latencies):.3f} ms max={max(kernel_latencies):.3f} ms "
                f"RDMA BW~={kernel_bw:.2f} GB/s (trace={trace_path})",
                flush=True,
            )
        except RuntimeError as e:
            print(f"[rank {rank}] profiler kernel extraction failed: {e}", flush=True)

    wall_bw = bytes_per_iter / 1e6 / wall_ms
    print(
        f"[rank {rank}] end2end: avg={wall_ms:.3f} ms/iter over {timed_iters} iters "
        f"RDMA BW~={wall_bw:.2f} GB/s (incl. host overhead)",
        flush=True,
    )

    try:
        torch.ops.symm_mem.internode_dispatch_rdma_sender_finalize(
            torch.empty(0, device=device)
        )
    except Exception as e:
        print(f"[rank {rank}] finalize raised: {e!r}", flush=True)
    sys.stdout.flush()
    # Skip dist.destroy_process_group()/an extra barrier here: correctness has
    # already been verified above and this repo's xccl backend can hang on a
    # trailing barrier under narrow ZE_AFFINITY_MASK per-rank device binding
    # (an environment quirk unrelated to this op) -- os._exit avoids waiting
    # on it so the test still reports PASS/FAIL promptly.
    os._exit(0)


if __name__ == "__main__":
    main()
