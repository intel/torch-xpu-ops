"""Minimal, reliable reproducer for the ~32 ms cross-device peer-access stall.

Root cause (localized in test_notify_dispatch_stall_probe.py): the ~32 ms
spike seen inside SymmBuffer.allgather_local_permute_fusion originates from the
notify_dispatch kernel's *broad peer symmetric-memory access* (it reads every
rank's slot via topk_rank_ptrs).  This script strips away notify_dispatch
entirely and reproduces the identical ~32 ms stall with nothing but a plain
device copy that reads ALL ranks' peer slots each iteration (remote_copy_all).

Local kernels, the symmetric-memory barrier, and single-neighbor peer reads do
NOT stall -- only the all-ranks peer fan-out does.

The stall is intermittent and time-correlated (a background/driver event that
periodically interferes with PCIe peer access), so the loop must run long
enough (several hundred ms of wall time) to contain at least one occurrence.

Usage:
    mpirun -np 8 python test_remote_copy_all_stall_repro.py
    #   REPRO_LOOP=3000      iterations (default 3000, ~13 s wall on 8x PVC)
    #   REPRO_SPIKE_MS=10     spike threshold in ms (default 10)
"""

import os

import torch
import torch.distributed as dist

import env
from symm_buffer import SymmBuffer

TOKENS_PER_RANK = env.tokens_per_rank()
HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
LOOP = int(os.environ.get("REPRO_LOOP", "3000"))
SPIKE_MS = float(os.environ.get("REPRO_SPIKE_MS", "10"))


def init_distributed():
    env.setup_distributed_env(master_addr="localhost", master_port="29535")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def main():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD
    ntr = TOKENS_PER_RANK

    sbuf = SymmBuffer(
        group=group, num_max_tokens_per_rank=ntr,
        hidden=HIDDEN_SIZE, num_topk=TOPK,
    )

    # Stage this rank's topk slot once so every rank's slot is populated, then
    # build peer views onto ALL ranks' topk slots (the notify access pattern).
    torch.manual_seed(1000 + rank)
    my_topk = torch.randint(0, NUM_EXPERTS, (ntr, TOPK), device=device, dtype=torch.int32)
    sbuf._topk_local_slot[:ntr, :TOPK].copy_(my_topk)
    sbuf.workspace.barrier()

    remote_all = []
    for r in range(world_size):
        off = sbuf._topk_base_offset + r * ntr * sbuf.num_topk
        remote_all.append(
            sbuf.workspace.get_buffer(r, (ntr, TOPK), torch.int32, storage_offset=off)
        )
    dst = torch.empty(world_size, ntr, TOPK, device=device, dtype=torch.int32)

    if rank == 0:
        print(f"[REPRO] remote_copy_all: world_size={world_size} "
              f"tokens_per_rank={ntr} loop={LOOP} spike_ms={SPIKE_MS}", flush=True)

    begin = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    torch.xpu.synchronize()
    dist.barrier()
    for i in range(LOOP):
        begin[i].record()
        for r in range(world_size):
            dst[r].copy_(remote_all[r])
        end[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    lat = [begin[i].elapsed_time(end[i]) for i in range(LOOP)]
    spikes = [(i, lat[i]) for i in range(LOOP) if lat[i] >= SPIKE_MS]

    # Reduce reproduction status across ranks: reproduced if ANY rank stalled.
    n_spikes = torch.tensor([len(spikes)], device=device)
    dist.all_reduce(n_spikes, op=dist.ReduceOp.SUM)
    max_lat = torch.tensor([max(lat)], device=device)
    dist.all_reduce(max_lat, op=dist.ReduceOp.MAX)

    s = sorted(lat)
    print(f"[Rank {rank}] median={s[len(s)//2]:.3f} ms  max={max(lat):.3f} ms  "
          f"spikes(>= {SPIKE_MS}ms)={len(spikes)}  "
          f"{[(i, round(v,1)) for i, v in spikes[:8]]}", flush=True)

    if rank == 0:
        reproduced = int(n_spikes.item()) > 0
        print("\n" + "=" * 60, flush=True)
        print(f"[REPRO] 32ms stall via remote_copy_all: "
              f"{'REPRODUCED' if reproduced else 'NOT reproduced'}  "
              f"(total_spikes={int(n_spikes.item())}, "
              f"global_max={max_lat.item():.1f} ms)", flush=True)
        if not reproduced:
            print("[REPRO] increase REPRO_LOOP (need enough wall time to hit the "
                  "periodic background event).", flush=True)
        print("=" * 60, flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
