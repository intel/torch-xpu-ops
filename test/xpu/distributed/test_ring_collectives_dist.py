"""
Accuracy check for ring_allgather and ring_reduce_scatter (symmetric memory).

Usage:
    mpirun -n 2 python test_ring_collectives_dist.py
    mpirun -n 4 python test_ring_collectives_dist.py
"""
import os

import torch
import torch.distributed as dist

from ring_collectives import (
    _HAS_RING_ALLGATHER,
    _HAS_RING_REDUCE_SCATTER,
    ring_allgather,
    ring_reduce_scatter,
)

CHUNK = 4096
DTYPE = torch.bfloat16


def _init():
    os.environ.setdefault("RANK", str(os.environ.get("PMI_RANK", 0)))
    os.environ.setdefault("WORLD_SIZE", str(os.environ.get("PMI_SIZE", 1)))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29801")
    dist.init_process_group("xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(f"xpu:{rank}")
    return rank, world_size


def test_ring_allgather(rank, world_size, device):
    torch.manual_seed(1234 + rank)
    shard = torch.randn(CHUNK, dtype=DTYPE, device=device)

    out = ring_allgather(shard).clone()

    gathered = [torch.empty(CHUNK, dtype=DTYPE, device=device) for _ in range(world_size)]
    dist.all_gather(gathered, shard)
    expected = torch.cat(gathered, dim=0)

    torch.xpu.synchronize()
    ok = torch.equal(out, expected)
    max_err = (out.float() - expected.float()).abs().max().item()
    if rank == 0:
        print(f"[ring_allgather] match={ok} max_err={max_err}")
    return ok


def test_ring_reduce_scatter(rank, world_size, device):
    torch.manual_seed(4321 + rank)
    full = torch.randn(CHUNK * world_size, dtype=DTYPE, device=device)

    out = ring_reduce_scatter(full).clone()

    reduced = full.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    expected = reduced[rank * CHUNK : (rank + 1) * CHUNK]

    torch.xpu.synchronize()
    max_err = (out.float() - expected.float()).abs().max().item()
    # bf16 accumulation order differs; allow a small tolerance.
    tol = 1e-2 * expected.float().abs().max().clamp_min(1.0).item()
    ok = max_err <= tol
    if rank == 0:
        print(f"[ring_reduce_scatter] match={ok} max_err={max_err} tol={tol}")
    return ok


def main():
    rank, world_size = _init()
    device = f"xpu:{rank}"

    if not (_HAS_RING_ALLGATHER and _HAS_RING_REDUCE_SCATTER):
        if rank == 0:
            print("Native ring kernels not built; run csrc/build.py first.")
        dist.destroy_process_group()
        return

    ag_ok = test_ring_allgather(rank, world_size, device)
    rs_ok = test_ring_reduce_scatter(rank, world_size, device)

    dist.barrier()
    if rank == 0:
        status = "PASS" if (ag_ok and rs_ok) else "FAIL"
        print(f"=== ring collectives: {status} ===")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
