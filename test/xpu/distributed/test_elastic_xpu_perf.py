"""
Performance benchmark for ElasticBuffer dispatch and combine APIs.

Measures GPU kernel latency with event-based timing and prints
bandwidth-based projections for each rank.

Usage:
    mpirun -n 4 python test_elastic_xpu_perf.py
"""

import os

import torch
import torch.distributed as dist

from allgather_local_permute_fusion import compute_scatter_idx
from deepep_dispatch import get_expert_owner
from elastic_xpu import ElasticBuffer

TOKENS_PER_RANK = 4096
HIDDEN_SIZE = 5120
TOPK = 8
NUM_EXPERTS = 128
LOOP = 40
WARMUP = 20
PCIE_DISCOUNT = 0.7
CROSS_GPU_BW_GBPS = 31.5 * PCIE_DISCOUNT
HBM_BW_GBPS = 437.0


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29523"
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def make_inputs(rank, world_size, device):
    num_tokens_per_rank = TOKENS_PER_RANK
    num_tokens = num_tokens_per_rank * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        num_tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )

    # Generate global topk_idx (deterministic across ranks), then take local slice.
    torch.manual_seed(42)
    global_topk_idx = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOPK), device=device, dtype=torch.int32
    )
    topk_idx = global_topk_idx[rank * num_tokens_per_rank : (rank + 1) * num_tokens_per_rank].clone()

    torch.manual_seed(777)
    global_topk_weights = torch.rand(num_tokens, TOPK, device=device, dtype=torch.float32)
    global_topk_weights = global_topk_weights / global_topk_weights.sum(dim=1, keepdim=True)
    topk_weights = global_topk_weights[rank * num_tokens_per_rank : (rank + 1) * num_tokens_per_rank].clone()

    return hidden_shard, topk_idx, global_topk_idx, topk_weights


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def _build_owner_table(topk_idx, num_experts, world_size):
    owner_table = torch.tensor(
        [get_expert_owner(e, num_experts, world_size) for e in range(num_experts)],
        device=topk_idx.device,
        dtype=torch.int64,
    )
    return owner_table[topk_idx.to(torch.int64)]


def build_dispatch_projection(topk_idx, num_tokens_per_rank, hidden_size,
                              elem_size, num_experts, world_size):
    """Per-rank projection for dispatch:
    1. Remote fetch – PCIe read time for unique tokens from other ranks.
    2. Topk write  – HBM write time for owned (token, k) rows.
    """
    num_tokens = topk_idx.shape[0]
    bytes_per_token = hidden_size * elem_size
    owners_2d = _build_owner_table(topk_idx, num_experts, world_size)
    token_ids = torch.arange(num_tokens, device=topk_idx.device, dtype=torch.int64)
    src_ranks = token_ids // num_tokens_per_rank

    remote_fetch_ms, topk_write_ms = [], []
    for owner_rank in range(world_size):
        has_owned = (owners_2d == owner_rank).any(dim=1)
        matching_src = src_ranks[has_owned]
        remote_count = (matching_src != owner_rank).sum().item()
        fetch_ms = remote_count * bytes_per_token / (CROSS_GPU_BW_GBPS * 1e9) * 1e3
        remote_fetch_ms.append(fetch_ms)

        owned_assignments = (owners_2d == owner_rank).sum().item()
        write_ms = owned_assignments * bytes_per_token / (HBM_BW_GBPS * 1e9) * 1e3
        topk_write_ms.append(write_ms)

    return remote_fetch_ms, topk_write_ms


def build_combine_projection(topk_idx, num_tokens_per_rank, hidden_size,
                             elem_size, num_experts, world_size):
    """Per-rank projection for combine:
    1. Expert read  – HBM read time for owned expert_output rows.
    2. Remote push  – PCIe write time for partial sums to non-local target ranks.
    3. Local reduce – HBM write time for summing received partials into output.
    """
    num_tokens, topk = topk_idx.shape
    bytes_per_token = hidden_size * elem_size
    owners_2d = _build_owner_table(topk_idx, num_experts, world_size)
    token_ids = torch.arange(num_tokens, device=topk_idx.device, dtype=torch.int64)
    target_ranks = token_ids // num_tokens_per_rank

    expert_read_ms, remote_push_ms, local_reduce_ms = [], [], []
    for owner_rank in range(world_size):
        owned_mask = owners_2d == owner_rank
        owned_count = owned_mask.sum().item()
        read_ms = owned_count * bytes_per_token / (HBM_BW_GBPS * 1e9) * 1e3
        expert_read_ms.append(read_ms)

        remote_targets = set()
        for tok in range(num_tokens):
            for k in range(topk):
                if int(owned_mask[tok, k].item()):
                    tgt = int(target_ranks[tok].item())
                    if tgt != owner_rank:
                        remote_targets.add(tgt)

        remote_tokens_per_target = {}
        for tok in range(num_tokens):
            tgt = int(target_ranks[tok].item())
            if tgt != owner_rank and tgt in remote_targets:
                if tgt not in remote_tokens_per_target:
                    remote_tokens_per_target[tgt] = 0
                if owned_mask[tok].any().item():
                    remote_tokens_per_target[tgt] += 1

        remote_bytes = sum(remote_tokens_per_target.values()) * bytes_per_token
        push_ms = remote_bytes / (CROSS_GPU_BW_GBPS * 1e9) * 1e3
        remote_push_ms.append(push_ms)

        reduce_ms = num_tokens_per_rank * bytes_per_token / (HBM_BW_GBPS * 1e9) * 1e3
        local_reduce_ms.append(reduce_ms)

    return expert_read_ms, remote_push_ms, local_reduce_ms


# ---------------------------------------------------------------------------
# Dispatch performance benchmark
# ---------------------------------------------------------------------------

def bench_elastic_dispatch():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    hidden_shard, topk_idx, global_topk_idx, topk_weights = make_inputs(rank, world_size, device)

    buffer = ElasticBuffer(
        group=group,
        num_max_tokens_per_rank=TOKENS_PER_RANK,
        hidden=HIDDEN_SIZE,
        num_topk=TOPK,
    )

    # Timed loop (first WARMUP iterations are not recorded)
    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    for i in range(LOOP):
        if i >= WARMUP:
            begin_events[i].record()
        buffer.dispatch(
            hidden_shard,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=NUM_EXPERTS,
            do_cpu_sync=False,
        )
        if i >= WARMUP:
            end_events[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    avg = sum(latencies) / len(latencies)
    print(f"[ElasticBuffer.dispatch  latency rank {rank}] {latencies} ms")

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"[Dispatch] avg={avg:.3f} ms  min={min(latencies):.3f} ms  max={max(latencies):.3f} ms")

        # Projections use the global topk_idx for full traffic analysis.
        remote_fetch_ms, topk_write_ms = build_dispatch_projection(
            topk_idx=global_topk_idx,
            num_tokens_per_rank=TOKENS_PER_RANK,
            hidden_size=HIDDEN_SIZE,
            elem_size=hidden_shard.element_size(),
            num_experts=NUM_EXPERTS,
            world_size=world_size,
        )
        print(f"\n[Dispatch Projection] PCIe BW={CROSS_GPU_BW_GBPS:.1f} GB/s, HBM BW={HBM_BW_GBPS:.1f} GB/s")
        for r in range(world_size):
            total_ms = remote_fetch_ms[r] + topk_write_ms[r]
            print(
                f"  rank {r}: remote_fetch={remote_fetch_ms[r]:.3f} ms, "
                f"topk_write={topk_write_ms[r]:.3f} ms, total={total_ms:.3f} ms"
            )
        worst = max(range(world_size), key=lambda r: remote_fetch_ms[r] + topk_write_ms[r])
        print(
            f"  worst_rank={worst}, projected_total={remote_fetch_ms[worst] + topk_write_ms[worst]:.3f} ms"
        )
        print(f"{'='*65}\n")

    return buffer, hidden_shard, topk_idx, global_topk_idx, topk_weights


# ---------------------------------------------------------------------------
# Combine performance benchmark
# ---------------------------------------------------------------------------

def bench_elastic_combine(buffer, hidden_shard, topk_idx, global_topk_idx, topk_weights):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"xpu:{rank}"

    # Run one dispatch to get a valid handle
    dispatch_out, _, recv_topk_weights, handle, _ = buffer.dispatch(
        hidden_shard,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=NUM_EXPERTS,
        do_cpu_sync=True,
    )
    torch.xpu.synchronize()
    dist.barrier()

    # The notify_dispatch kernel has a non-deterministic race condition in its
    # barrier synchronization that can produce out-of-bounds scatter_idx.
    # Workaround: compute the correct scatter_idx deterministically and replace
    # the kernel's output.  This does not affect combine performance numbers
    # since the kernel workload is identical regardless of scatter_idx values.
    correct_scatter_idx, _ = compute_scatter_idx(global_topk_idx, num_experts=NUM_EXPERTS)
    handle.scatter_idx = correct_scatter_idx

    num_tokens = TOKENS_PER_RANK * world_size
    expert_output = torch.randn(
        num_tokens * TOPK, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    )

    # Separate warmup phase
    for _ in range(WARMUP):
        buffer.combine(
            expert_output,
            handle=handle,
            topk_weights=recv_topk_weights,
        )
    torch.xpu.synchronize()
    dist.barrier()

    # Timed loop (first WARMUP iterations are not recorded)
    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    for i in range(LOOP):
        if i >= WARMUP:
            begin_events[i].record()
        buffer.combine(
            expert_output,
            handle=handle,
            topk_weights=recv_topk_weights,
        )
        if i >= WARMUP:
            end_events[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    latencies = [begin_events[i].elapsed_time(end_events[i]) for i in range(WARMUP, LOOP)]
    avg = sum(latencies) / len(latencies)
    print(f"[ElasticBuffer.combine   latency rank {rank}] {latencies} ms")

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"[Combine]  avg={avg:.3f} ms  min={min(latencies):.3f} ms  max={max(latencies):.3f} ms")

        # Projections use the global topk_idx for full traffic analysis.
        expert_read_ms, remote_push_ms, local_reduce_ms = build_combine_projection(
            topk_idx=global_topk_idx,
            num_tokens_per_rank=TOKENS_PER_RANK,
            hidden_size=HIDDEN_SIZE,
            elem_size=expert_output.element_size(),
            num_experts=NUM_EXPERTS,
            world_size=world_size,
        )
        print(f"\n[Combine Projection] PCIe BW={CROSS_GPU_BW_GBPS:.1f} GB/s, HBM BW={HBM_BW_GBPS:.1f} GB/s")
        for r in range(world_size):
            total_ms = expert_read_ms[r] + remote_push_ms[r] + local_reduce_ms[r]
            print(
                f"  rank {r}: expert_read={expert_read_ms[r]:.3f} ms, "
                f"remote_push={remote_push_ms[r]:.3f} ms, "
                f"local_reduce={local_reduce_ms[r]:.3f} ms, total={total_ms:.3f} ms"
            )
        worst = max(
            range(world_size),
            key=lambda r: expert_read_ms[r] + remote_push_ms[r] + local_reduce_ms[r],
        )
        worst_total = expert_read_ms[worst] + remote_push_ms[worst] + local_reduce_ms[worst]
        print(f"  worst_rank={worst}, projected_total={worst_total:.3f} ms")
        print(f"{'='*65}\n")


def main():
    buffer, hidden_shard, topk_idx, global_topk_idx, topk_weights = bench_elastic_dispatch()

    # Let GPU settle between benchmarks
    torch.xpu.synchronize()
    dist.barrier()
    torch.xpu.empty_cache()

    bench_elastic_combine(buffer, hidden_shard, topk_idx, global_topk_idx, topk_weights)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
