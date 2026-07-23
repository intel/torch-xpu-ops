"""Standalone UT: isolate notify_dispatch and probe for periodic latency stalls.

Background
----------
In test_symm_buffer_fusion_perf.py (np=8) the fused allgather_local_permute
path shows a recurring ~34 ms spike at iteration 0 AND iteration ~25.  The
per-op breakdown attributed the spike mostly to the ring op, but occasionally
to notify_dispatch on a single rank, which is the classic signature of a
*global* stall being charged to whichever collective a rank happens to block
on.

This UT removes the ring op entirely and drives ONLY notify_dispatch in a
tight loop, timing each call per-iteration with XPU events.  It then:
  * prints the full per-iteration latency list for every rank,
  * flags any iteration whose latency exceeds a spike threshold,
  * gathers all ranks' timings on rank 0 and prints an aligned table for the
    spike iterations so we can see whether the stall is global (all ranks slow
    at the same iteration) or local to notify_dispatch on one rank.

Usage:
    mpirun -np 8 python test_notify_dispatch_stall_ut.py
    # knobs:
    #   NOTIFY_UT_LOOP=200       total iterations (default 120)
    #   NOTIFY_UT_WARMUP=0       iterations excluded from stats (default 0)
    #   NOTIFY_UT_ABS=1          use notify_dispatch_v2_abs when available
    #   NOTIFY_UT_SPIKE_MS=10    spike threshold in ms (default 3x median)
"""

import os

import torch
import torch.distributed as dist

import env
from symm_buffer import SymmBuffer, _HAS_NOTIFY_DISPATCH_V2_ABS

TOKENS_PER_RANK = env.tokens_per_rank()
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
LOOP = int(os.environ.get("NOTIFY_UT_LOOP", "120"))
WARMUP = int(os.environ.get("NOTIFY_UT_WARMUP", "0"))
USE_ABS = os.environ.get("NOTIFY_UT_ABS", "1") == "1" and _HAS_NOTIFY_DISPATCH_V2_ABS


def init_distributed():
    env.setup_distributed_env(master_addr="localhost", master_port="29533")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def make_inputs(rank, world_size, device):
    num_tokens_per_rank = TOKENS_PER_RANK
    num_tokens = num_tokens_per_rank * world_size

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

    return topk_idx.contiguous(), topk_weights.contiguous()


def notify_only(sbuf, topk_idx, topk_weights, num_experts):
    """Run exactly the notify_dispatch portion of the ring dispatch path.

    Mirrors SymmBuffer._allgather_local_permute_fusion_ring: stage this rank's
    topk_idx / weights into symmetric memory, barrier, then invoke the
    notify_dispatch kernel. The ring op is intentionally omitted.
    """
    num_tokens_per_rank = topk_idx.shape[0]
    topk = topk_idx.shape[1]
    num_tokens = num_tokens_per_rank * sbuf.num_ranks

    sbuf._topk_local_slot[:num_tokens_per_rank, :topk].copy_(topk_idx)
    sbuf._weights_local_slot[:num_tokens_per_rank, :topk].copy_(topk_weights)
    sbuf.workspace.barrier()

    global_topk_idx = sbuf._global_topk_idx_buf[:num_tokens, :topk]
    global_topk_weights = sbuf._global_topk_weights_buf[:num_tokens, :topk]
    scatter_idx = sbuf._scatter_idx[:num_tokens, :topk]
    rows_per_expert = sbuf._rows_per_expert_buf[:num_experts]

    if USE_ABS:
        abs_scatter_idx = sbuf._abs_scatter_idx_buf[:num_tokens, :topk]
        torch.ops.symm_mem.notify_dispatch_v2_abs(
            sbuf._topk_rank_ptrs,
            global_topk_idx,
            scatter_idx,
            rows_per_expert,
            abs_scatter_idx,
            num_tokens_per_rank,
            topk,
            sbuf.num_topk,
            num_experts,
            sbuf.rank_idx,
            sbuf.num_ranks,
            sbuf._weights_rank_ptrs,
            global_topk_weights,
        )
    else:
        torch.ops.symm_mem.notify_dispatch_v2(
            sbuf._topk_rank_ptrs,
            global_topk_idx,
            scatter_idx,
            rows_per_expert,
            num_tokens_per_rank,
            topk,
            sbuf.num_topk,
            num_experts,
            sbuf.rank_idx,
            sbuf.num_ranks,
            sbuf._weights_rank_ptrs,
            global_topk_weights,
        )


def main():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    topk_idx, topk_weights = make_inputs(rank, world_size, device)

    sbuf = SymmBuffer(
        group=group,
        num_max_tokens_per_rank=TOKENS_PER_RANK,
        hidden=HIDDEN_SIZE,
        num_topk=TOPK,
    )

    op_name = "notify_dispatch_v2_abs" if USE_ABS else "notify_dispatch_v2"
    if rank == 0:
        print(
            f"[UT] isolating {op_name}: world_size={world_size} "
            f"tokens_per_rank={TOKENS_PER_RANK} topk={TOPK} "
            f"num_experts={NUM_EXPERTS} loop={LOOP} warmup={WARMUP}",
            flush=True,
        )

    begin = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    for i in range(LOOP):
        begin[i].record()
        notify_only(sbuf, topk_idx, topk_weights, NUM_EXPERTS)
        end[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    lat = [begin[i].elapsed_time(end[i]) for i in range(LOOP)]
    timed = lat[WARMUP:]

    sorted_timed = sorted(timed)
    median = sorted_timed[len(sorted_timed) // 2]
    spike_ms = float(os.environ.get("NOTIFY_UT_SPIKE_MS", str(max(3.0 * median, 5.0))))
    spikes = [i for i in range(LOOP) if lat[i] >= spike_ms]

    print(
        f"[Rank {rank}] {op_name} per-iter (ms): "
        f"{[f'{x:.3f}' for x in lat]}",
        flush=True,
    )
    print(
        f"[Rank {rank}] {op_name} stats: "
        f"median={median:.3f} avg={sum(timed)/len(timed):.3f} "
        f"min={min(timed):.3f} max={max(timed):.3f} "
        f"spike_threshold={spike_ms:.3f} "
        f"spike_iters={spikes}",
        flush=True,
    )

    # Gather every rank's per-iter timings on rank 0 and print an aligned table
    # for the union of spike iterations so we can see if stalls are global.
    all_lat = [None] * world_size
    all_spikes = [None] * world_size
    dist.all_gather_object(all_lat, lat)
    dist.all_gather_object(all_spikes, spikes)

    if rank == 0:
        union = sorted(set(i for s in all_spikes for i in s))
        print("\n" + "=" * 70, flush=True)
        print(f"[UT] {op_name} spike alignment across ranks", flush=True)
        if not union:
            print("  no spikes detected above threshold", flush=True)
        else:
            header = "  iter | " + " | ".join(f"r{r:>2}" for r in range(world_size))
            print(header, flush=True)
            print("  " + "-" * (len(header) - 2), flush=True)
            for it in union:
                row = " | ".join(f"{all_lat[r][it]:5.1f}" for r in range(world_size))
                n_slow = sum(1 for r in range(world_size) if all_lat[r][it] >= spike_ms)
                print(f"  {it:>4} | {row}   ({n_slow}/{world_size} ranks slow)", flush=True)
            print(
                "\n  Interpretation: if a spike row shows ALL ranks slow at the "
                "same iter,\n  the stall is global (sync/driver), not intrinsic "
                "to notify_dispatch.",
                flush=True,
            )
            if len(union) > 1:
                diffs = [union[k + 1] - union[k] for k in range(len(union) - 1)]
                print(f"  spike iteration gaps: {diffs}", flush=True)
        print("=" * 70 + "\n", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
