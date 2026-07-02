"""
Performance benchmark for SymmBuffer fusion APIs on XPU.

The script runs three groups for each fused op:
1. FUSION_RING=0
2. FUSION_RING=1
3. benchmark_ring reference path

The first two groups are compared for accuracy, then the benchmark_ring
reference is reported as a third performance point.

Usage:
    mpirun -n 2 python test_symm_buffer_fusion_perf.py
"""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

import env

import symm_buffer as symm_buffer_mod
from allgather_local_permute_fusion import _HAS_LOCAL_PERMUTE_KERNEL, compute_scatter_idx
from unpermute_reducescatter_fusion import _HAS_LOCAL_UNPERMUTE_KERNEL

TOKENS_PER_RANK = env.tokens_per_rank()
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
LOOP = 40
WARMUP = 20


def init_distributed():
    env.setup_distributed_env(master_addr="localhost", master_port="29525")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


@contextmanager
def set_module_fusion_ring(enabled):
    previous = symm_buffer_mod._FUSION_RING
    symm_buffer_mod._FUSION_RING = enabled
    try:
        yield
    finally:
        symm_buffer_mod._FUSION_RING = previous


def create_symm_buffer(enabled, group):
    with set_module_fusion_ring(enabled):
        sbuf = symm_buffer_mod.SymmBuffer(
            group=group,
            num_max_tokens_per_rank=TOKENS_PER_RANK,
            hidden=HIDDEN_SIZE,
            num_topk=TOPK,
        )
    if enabled and not getattr(sbuf, "_fusion_ring", False):
        raise RuntimeError("FUSION_RING=1 requested, but the ring backend is unavailable")
    return sbuf


def make_inputs(rank, world_size, device):
    num_tokens_per_rank = TOKENS_PER_RANK
    num_tokens = num_tokens_per_rank * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        num_tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )

    torch.manual_seed(42)
    global_topk_idx = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOPK), device=device, dtype=torch.int32
    )
    topk_idx = global_topk_idx[
        rank * num_tokens_per_rank : (rank + 1) * num_tokens_per_rank
    ].clone()

    torch.manual_seed(777)
    global_topk_weights = torch.rand(
        num_tokens, TOPK, device=device, dtype=torch.float32
    )
    global_topk_weights = global_topk_weights / global_topk_weights.sum(
        dim=1, keepdim=True
    )
    topk_weights = global_topk_weights[
        rank * num_tokens_per_rank : (rank + 1) * num_tokens_per_rank
    ].clone()

    return hidden_shard, topk_idx, global_topk_idx, topk_weights, global_topk_weights


def timed_loop(fn, loop, warmup):
    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    for i in range(loop):
        if i >= warmup:
            begin_events[i].record()
        fn()
        if i >= warmup:
            end_events[i].record()
    torch.xpu.synchronize()
    dist.barrier()
    return [begin_events[i].elapsed_time(end_events[i]) for i in range(warmup, loop)]


def print_latency_summary(label, latencies, rank):
    avg = sum(latencies) / len(latencies)
    print(f"[{label} latency rank {rank}] {[f'{l:.3f}' for l in latencies]} ms")
    return avg


def build_allgather_local_permute_reference(hidden_shard, scatter_idx, group):
    world_size = dist.get_world_size(group)
    num_tokens_per_rank, hidden = hidden_shard.shape
    num_tokens, topk = scatter_idx.shape
    gathered = torch.empty(num_tokens, hidden, device=hidden_shard.device, dtype=hidden_shard.dtype)
    out = torch.empty(
        num_tokens * topk, hidden, device=hidden_shard.device, dtype=hidden_shard.dtype
    )

    def run_reference():
        dist.all_gather_into_tensor(gathered, hidden_shard, group=group)
        torch.ops.symm_mem.local_permute_copy_(
            gathered, scatter_idx.contiguous(), 0, out
        )

    latencies = timed_loop(run_reference, LOOP, WARMUP)
    return out, latencies


def build_unpermute_reducescatter_reference(
    expert_output,
    scatter_idx,
    topk_weights,
    num_tokens_per_rank,
    group,
):
    num_tokens, hidden = scatter_idx.shape[0], expert_output.shape[1]
    full_result = torch.empty(num_tokens, hidden, device=expert_output.device, dtype=expert_output.dtype)
    out = torch.empty(num_tokens_per_rank, hidden, device=expert_output.device, dtype=expert_output.dtype)

    def run_reference():
        torch.ops.symm_mem.local_unpermute_copy_(
            expert_output,
            scatter_idx.contiguous(),
            topk_weights.contiguous(),
            0,
            num_tokens,
            full_result,
        )
        dist.reduce_scatter_tensor(out, full_result, group=group)

    latencies = timed_loop(run_reference, LOOP, WARMUP)
    return out, latencies


def benchmark_allgather_mode(
    sbuf,
    fusion_ring_label,
    hidden_shard,
    topk_idx,
    topk_weights,
    scatter_idx,
    num_experts,
    rank,
    world_size,
    device,
):
    num_tokens_per_rank = TOKENS_PER_RANK
    num_tokens = num_tokens_per_rank * world_size
    group = dist.group.WORLD
    remap = torch.empty((num_tokens * TOPK, HIDDEN_SIZE), device=device, dtype=torch.bfloat16)
    last_handle = None

    def run_mode():
        nonlocal last_handle
        _, last_handle = sbuf.allgather_local_permute_fusion(
            hidden_shard=hidden_shard,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=num_experts,
            remap_hidden_states=remap,
        )

    latencies = timed_loop(run_mode, LOOP, WARMUP)
    print_latency_summary(f"{fusion_ring_label} allgather_permute", latencies, rank)
    return remap, last_handle, latencies


def benchmark_unpermute_mode(
    sbuf,
    fusion_ring_label,
    expert_output,
    scatter_idx,
    global_topk_weights,
    handle,
    rank,
    world_size,
    device,
):
    num_tokens_per_rank = TOKENS_PER_RANK
    output = torch.empty(num_tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    def run_mode():
        sbuf.unpermute_reducescatter_fusion(
            expert_output=expert_output,
            handle=handle,
            output=output,
        )

    latencies = timed_loop(run_mode, LOOP, WARMUP)
    print_latency_summary(f"{fusion_ring_label} unpermute_reducescatter", latencies, rank)
    return output, latencies


def _canonical_permuted(remap, abs_scatter_idx):
    """Reconstruct the permuted data in canonical (token, top-k) order.

    The permuted ``remap`` buffer groups tokens by expert, and the row a token
    lands on within its expert bucket is assigned non-deterministically by
    ``notify_dispatch_v2`` (work-groups race to reserve position chunks). Two
    independent runs therefore produce different—but equally valid—permutations,
    so comparing the raw buffers row-by-row is meaningless. Gathering each row
    back through the path's own ``abs_scatter_idx`` yields a layout that is
    invariant to the within-expert ordering, so two correct paths match exactly.
    """
    flat = abs_scatter_idx.reshape(-1)
    valid = flat >= 0
    out = torch.zeros(flat.numel(), remap.shape[1], device=remap.device, dtype=remap.dtype)
    out[valid] = remap[flat[valid].long()]
    return out


def report_allgather_accuracy(rank, fusion0, handle0, fusion1, handle1):
    canon0 = _canonical_permuted(fusion0, handle0.abs_scatter_idx)
    canon1 = _canonical_permuted(fusion1, handle1.abs_scatter_idx)
    match = torch.equal(canon0, canon1)
    max_diff = (canon0.float() - canon1.float()).abs().max().item()
    print(
        f"[Rank {rank}] allgather_permute accuracy FUSION_RING=0 vs FUSION_RING=1 "
        f"(order-independent): match={match} max_diff={max_diff:.6f}"
    )
    assert match, (
        f"allgather_permute mismatch on rank {rank}: max_diff={max_diff:.6f}"
    )


def _build_consistent_expert_output(canonical_src, abs_scatter_idx):
    """Scatter per-(token, top-k) canonical values into the permuted-row layout
    using a path's own ``abs_scatter_idx``.

    Each fusion path assigns a different (but valid) within-expert ordering, so a
    single shared random ``expert_output`` would be read through different row
    indices by each path and yield divergent combine results. Building the
    expert output by scattering the SAME canonical per-(token, top-k) data
    through each path's own mapping guarantees that the row every path reads back
    during unpermute holds identical logical data (exactly what a real expert
    stage would produce), making the per-token outputs comparable.
    """
    eo = torch.zeros_like(canonical_src)
    flat = abs_scatter_idx.reshape(-1)
    valid = flat >= 0
    eo[flat[valid].long()] = canonical_src[valid]
    return eo


def report_unpermute_accuracy(rank, fusion0, fusion1, world_size):
    atol = 0.025 * world_size + 0.01
    match = torch.allclose(fusion0, fusion1, atol=atol, rtol=1e-2)
    max_diff = (fusion0 - fusion1).abs().max().item()
    print(
        f"[Rank {rank}] unpermute_reducescatter accuracy FUSION_RING=0 vs FUSION_RING=1: "
        f"match={match} max_diff={max_diff:.6f} atol={atol}"
    )
    assert match, (
        f"unpermute_reducescatter mismatch on rank {rank}: max_diff={max_diff:.6f}, atol={atol}"
    )


def main():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    hidden_shard, topk_idx, global_topk_idx, topk_weights, global_topk_weights = make_inputs(
        rank, world_size, device
    )
    scatter_idx, _ = compute_scatter_idx(global_topk_idx, num_experts=NUM_EXPERTS)

    if not getattr(symm_buffer_mod, "_HAS_NOTIFY_DISPATCH_V2_KERNEL", False):
        raise RuntimeError("notify_dispatch_v2 kernel is required for SymmBuffer fusion benchmarks")

    if not _HAS_LOCAL_PERMUTE_KERNEL:
        raise RuntimeError("local_permute_copy_ kernel is required for benchmark_ring allgather reference")
    if not _HAS_LOCAL_UNPERMUTE_KERNEL:
        raise RuntimeError("local_unpermute_copy_ kernel is required for benchmark_ring unpermute reference")

    sbuf_fusion0 = create_symm_buffer(False, group)
    sbuf_fusion1 = create_symm_buffer(True, group)

    print(f"\n{'=' * 70}")
    print("[Allgather + Permute Fusion]")

    fusion0_remap, fusion0_handle, fusion0_latencies = benchmark_allgather_mode(
        sbuf_fusion0,
        "FUSION_RING=0",
        hidden_shard,
        topk_idx,
        topk_weights,
        scatter_idx,
        NUM_EXPERTS,
        rank,
        world_size,
        device,
    )
    fusion1_remap, fusion1_handle, fusion1_latencies = benchmark_allgather_mode(
        sbuf_fusion1,
        "FUSION_RING=1",
        hidden_shard,
        topk_idx,
        topk_weights,
        scatter_idx,
        NUM_EXPERTS,
        rank,
        world_size,
        device,
    )
    report_allgather_accuracy(rank, fusion0_remap, fusion0_handle, fusion1_remap, fusion1_handle)

    ref_remap, ref_latencies = build_allgather_local_permute_reference(
        hidden_shard, scatter_idx, group
    )
    print_latency_summary("benchmark_ring reference allgather_permute", ref_latencies, rank)
    # Compare against the reference order-independently and only over the top-k
    # slots this rank actually owns (the fused path fills owned-expert rows only,
    # whereas the reference permutes every token to every expert).
    canon_f1 = _canonical_permuted(fusion1_remap, fusion1_handle.abs_scatter_idx)
    canon_ref = _canonical_permuted(ref_remap, scatter_idx)
    owned = fusion1_handle.abs_scatter_idx.reshape(-1) >= 0
    ref_match = torch.equal(canon_f1[owned], canon_ref[owned])
    print(
        f"[Rank {rank}] benchmark_ring reference allgather_permute vs FUSION_RING=1 "
        f"(order-independent): match={ref_match} "
        f"max_diff={(canon_f1[owned].float() - canon_ref[owned].float()).abs().max().item():.6f}"
    )

    torch.xpu.synchronize()
    dist.barrier()
    torch.xpu.empty_cache()

    torch.manual_seed(1000)
    canonical_src = torch.randn(
        world_size * TOKENS_PER_RANK * TOPK, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    # Each path reads its expert output through its own permutation, so build a
    # per-path expert_output that holds the same canonical per-(token, top-k)
    # data at each path's own rows (see _build_consistent_expert_output).
    expert_output0 = _build_consistent_expert_output(canonical_src, fusion0_handle.abs_scatter_idx)
    expert_output1 = _build_consistent_expert_output(canonical_src, fusion1_handle.abs_scatter_idx)
    expert_output_ref = _build_consistent_expert_output(canonical_src, scatter_idx)

    print(f"\n{'=' * 70}")
    print("[Unpermute + Reduce-Scatter Fusion]")

    fusion0_output, fusion0_unperm_latencies = benchmark_unpermute_mode(
        sbuf_fusion0,
        "FUSION_RING=0",
        expert_output0,
        fusion0_handle.abs_scatter_idx,
        fusion0_handle.global_topk_weights,
        fusion0_handle,
        rank,
        world_size,
        device,
    )
    fusion1_output, fusion1_unperm_latencies = benchmark_unpermute_mode(
        sbuf_fusion1,
        "FUSION_RING=1",
        expert_output1,
        fusion1_handle.abs_scatter_idx,
        fusion1_handle.global_topk_weights,
        fusion1_handle,
        rank,
        world_size,
        device,
    )
    report_unpermute_accuracy(rank, fusion0_output, fusion1_output, world_size)

    ref_output, ref_unperm_latencies = build_unpermute_reducescatter_reference(
        expert_output_ref,
        scatter_idx,
        global_topk_weights,
        TOKENS_PER_RANK,
        group,
    )
    print_latency_summary("benchmark_ring reference unpermute_reducescatter", ref_unperm_latencies, rank)
    ref_unperm_match = torch.allclose(fusion1_output, ref_output, atol=0.025 * world_size + 0.01, rtol=1e-2)
    print(
        f"[Rank {rank}] benchmark_ring reference unpermute_reducescatter vs FUSION_RING=1: "
        f"match={ref_unperm_match} max_diff={(fusion1_output - ref_output).abs().max().item():.6f}"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
