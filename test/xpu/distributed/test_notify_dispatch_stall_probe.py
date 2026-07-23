"""Diagnostic probe: which primitive causes the periodic ~32 ms notify stall?

Runs several loop "modes" under mpirun, each timing one isolated operation
per iteration with XPU events, then reports spike iterations and cross-rank
alignment for every mode.  By comparing which modes exhibit the periodic
~32 ms stall we can localize the root cause:

  barrier   : only sbuf.workspace.barrier()          (symmetric-mem sync)
  copy      : only a local device copy (no barrier, no cross-rank)
  notify_nb : notify kernel WITHOUT the preceding barrier
  notify    : barrier + notify kernel  (the real dispatch prologue)
  empty     : a trivial local elementwise kernel (driver/queue baseline)

Usage:
    mpirun -np 8 python test_notify_dispatch_stall_probe.py
    #   NOTIFY_UT_LOOP=200
    #   NOTIFY_UT_MODES=barrier,copy,notify_nb,notify,empty
    #   NOTIFY_UT_SPIKE_MS=5
"""

import os
import time

import torch
import torch.distributed as dist

import env
from symm_buffer import SymmBuffer, _HAS_NOTIFY_DISPATCH_V2_ABS

TOKENS_PER_RANK = env.tokens_per_rank()
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
LOOP = int(os.environ.get("NOTIFY_UT_LOOP", "150"))
SPIKE_MS = float(os.environ.get("NOTIFY_UT_SPIKE_MS", "5"))
PER_ITER_SYNC = os.environ.get("NOTIFY_UT_PER_ITER_SYNC", "0") == "1"
MODES = os.environ.get(
    "NOTIFY_UT_MODES",
    "empty,remote_copy,remote_copy_all,notify_nb",
).split(",")
USE_ABS = _HAS_NOTIFY_DISPATCH_V2_ABS


def init_distributed():
    env.setup_distributed_env(master_addr="localhost", master_port="29534")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def make_inputs(rank, world_size, device):
    ntr = TOKENS_PER_RANK
    num_tokens = ntr * world_size
    torch.manual_seed(42)
    gidx = torch.randint(0, NUM_EXPERTS, (num_tokens, TOPK), device=device, dtype=torch.int32)
    topk_idx = gidx[rank * ntr : (rank + 1) * ntr].clone().contiguous()
    torch.manual_seed(777)
    gw = torch.rand(num_tokens, TOPK, device=device, dtype=torch.float32)
    gw = gw / gw.sum(dim=1, keepdim=True)
    topk_weights = gw[rank * ntr : (rank + 1) * ntr].clone().contiguous()
    return topk_idx, topk_weights


def call_notify(sbuf, num_experts):
    ntr = TOKENS_PER_RANK
    topk = TOPK
    num_tokens = ntr * sbuf.num_ranks
    g_idx = sbuf._global_topk_idx_buf[:num_tokens, :topk]
    g_w = sbuf._global_topk_weights_buf[:num_tokens, :topk]
    sc = sbuf._scatter_idx[:num_tokens, :topk]
    rpe = sbuf._rows_per_expert_buf[:num_experts]
    if USE_ABS:
        absc = sbuf._abs_scatter_idx_buf[:num_tokens, :topk]
        torch.ops.symm_mem.notify_dispatch_v2_abs(
            sbuf._topk_rank_ptrs, g_idx, sc, rpe, absc,
            ntr, topk, sbuf.num_topk, num_experts,
            sbuf.rank_idx, sbuf.num_ranks, sbuf._weights_rank_ptrs, g_w,
        )
    else:
        torch.ops.symm_mem.notify_dispatch_v2(
            sbuf._topk_rank_ptrs, g_idx, sc, rpe,
            ntr, topk, sbuf.num_topk, num_experts,
            sbuf.rank_idx, sbuf.num_ranks, sbuf._weights_rank_ptrs, g_w,
        )


def run_mode(mode, sbuf, topk_idx, topk_weights, scratch, remote_buf, remote_dst,
             heavy, remote_all, remote_all_dst):
    ntr = TOKENS_PER_RANK
    topk = TOPK
    # Prime staged inputs once (kept constant across iterations).
    sbuf._topk_local_slot[:ntr, :topk].copy_(topk_idx)
    sbuf._weights_local_slot[:ntr, :topk].copy_(topk_weights)
    sbuf.workspace.barrier()
    torch.xpu.synchronize()
    dist.barrier()

    begin = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    host_t = [0.0] * LOOP

    t0 = time.perf_counter()
    for i in range(LOOP):
        # PER_ITER_SYNC=1 makes host timestamps track device execution (used to
        # measure wall-clock spacing of stalls). Off by default so the loop
        # matches the real async workload where the ~32 ms stall is observed.
        if PER_ITER_SYNC:
            torch.xpu.synchronize()
        host_t[i] = (time.perf_counter() - t0) * 1e3  # ms
        begin[i].record()
        if mode == "empty":
            scratch.add_(1.0)
        elif mode == "copy":
            sbuf._topk_local_slot[:ntr, :topk].copy_(topk_idx)
        elif mode == "barrier":
            sbuf.workspace.barrier()
        elif mode == "heavy_local":
            heavy.add_(1.0)
        elif mode == "remote_copy":
            remote_dst.copy_(remote_buf)
        elif mode == "remote_copy_all":
            for r in range(sbuf.num_ranks):
                remote_all_dst[r].copy_(remote_all[r])
        elif mode == "notify_nb":
            call_notify(sbuf, NUM_EXPERTS)
        elif mode == "notify":
            sbuf.workspace.barrier()
            call_notify(sbuf, NUM_EXPERTS)
        else:
            raise ValueError(f"unknown mode {mode}")
        end[i].record()
    torch.xpu.synchronize()
    dist.barrier()
    lat = [begin[i].elapsed_time(end[i]) for i in range(LOOP)]
    return lat, host_t


def report(mode, lat, host_t, rank, world_size):
    s = sorted(lat)
    median = s[len(s) // 2]
    spikes = [i for i in range(LOOP) if lat[i] >= SPIKE_MS]
    all_lat = [None] * world_size
    all_spikes = [None] * world_size
    dist.all_gather_object(all_lat, lat)
    dist.all_gather_object(all_spikes, spikes)
    if rank != 0:
        return
    union = sorted(set(i for sp in all_spikes for i in sp))
    print(f"\n[MODE={mode}] median={median:.3f} ms  max={max(lat):.3f} ms  "
          f"spike_iters(any rank, >={SPIKE_MS}ms)={len(union)}", flush=True)
    if not union:
        print("  no spikes -> this primitive does NOT trigger the stall", flush=True)
        return
    for it in union:
        row = " ".join(f"{all_lat[r][it]:5.1f}" for r in range(world_size))
        n_slow = sum(1 for r in range(world_size) if all_lat[r][it] >= SPIKE_MS)
        print(f"  iter {it:>4} [{row}] ({n_slow}/{world_size} slow) "
              f"wall={host_t[it]:.1f}ms", flush=True)
    if len(union) > 1:
        wall_gaps = [round(host_t[union[k + 1]] - host_t[union[k]], 1)
                     for k in range(len(union) - 1)]
        print(f"  spike iter gaps ={[union[k+1]-union[k] for k in range(len(union)-1)]}",
              flush=True)
        print(f"  spike wall gaps(ms)={wall_gaps}", flush=True)


def main():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD
    topk_idx, topk_weights = make_inputs(rank, world_size, device)
    sbuf = SymmBuffer(
        group=group, num_max_tokens_per_rank=TOKENS_PER_RANK,
        hidden=HIDDEN_SIZE, num_topk=TOPK,
    )
    scratch = torch.zeros(1024, device=device, dtype=torch.float32)
    heavy = torch.zeros(4 * 1024 * 1024, device=device, dtype=torch.float32)

    # Peer read: pull a neighbor's own topk slot from its symmetric memory.
    ntr = TOKENS_PER_RANK
    neighbor = (rank + 1) % world_size
    neighbor_topk_off = sbuf._topk_base_offset + neighbor * ntr * sbuf.num_topk
    remote_buf = sbuf.workspace.get_buffer(
        neighbor, (ntr, TOPK), torch.int32, storage_offset=neighbor_topk_off,
    )
    remote_dst = torch.empty(ntr, TOPK, device=device, dtype=torch.int32)

    # Peer read fan-out to ALL ranks' topk slots (mirrors notify's access set).
    remote_all = []
    for r in range(world_size):
        off = sbuf._topk_base_offset + r * ntr * sbuf.num_topk
        remote_all.append(
            sbuf.workspace.get_buffer(r, (ntr, TOPK), torch.int32, storage_offset=off)
        )
    remote_all_dst = torch.empty(world_size, ntr, TOPK, device=device, dtype=torch.int32)

    if rank == 0:
        print(f"[PROBE] world_size={world_size} loop={LOOP} spike_ms={SPIKE_MS} "
              f"use_abs={USE_ABS} modes={MODES}", flush=True)

    for mode in MODES:
        lat, host_t = run_mode(
            mode, sbuf, topk_idx, topk_weights, scratch,
            remote_buf, remote_dst, heavy, remote_all, remote_all_dst,
        )
        report(mode, lat, host_t, rank, world_size)
        dist.barrier()

    if rank == 0:
        print("\n[PROBE] done. The first mode in the list that shows the periodic "
              "~32 ms\n        spike is the primitive responsible for the stall.",
              flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
