"""Component-level profiling of deepep_owner_combine internals."""
import os
import ctypes
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from allgather_local_permute_fusion import compute_scatter_idx
from deepep_dispatch import get_expert_owner, _HAS_EP_COMBINE_KERNEL

TOKENS_PER_RANK = 4096
HIDDEN = 7168
TOPK = 8
NUM_EXPERTS = 256
ITERS = 30


def main():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29521"
    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.xpu.set_device(rank)
    dev = f"xpu:{rank}"
    group = dist.group.WORLD
    gname = group.group_name

    ntok = TOKENS_PER_RANK * ws
    torch.manual_seed(42)
    topk_idx = torch.randint(0, NUM_EXPERTS, (ntok, TOPK), device=dev, dtype=torch.int32)
    scatter_idx, _ = compute_scatter_idx(topk_idx, num_experts=NUM_EXPERTS)
    torch.manual_seed(777)
    tw = torch.rand(ntok, TOPK, device=dev, dtype=torch.float32)
    tw = (tw / tw.sum(1, keepdim=True)).float()
    eo = torch.randn(ntok * TOPK, HIDDEN, device=dev, dtype=torch.bfloat16)

    ws_size = ws * TOKENS_PER_RANK * HIDDEN * eo.element_size()
    recv = symm_mem.get_symm_mem_workspace(gname, min_size=ws_size)
    my_buf = recv.get_buffer(rank, (ws, TOKENS_PER_RANK, HIDDEN), eo.dtype, storage_offset=0)
    ptrs = [recv.get_buffer(r, (ws, TOKENS_PER_RANK, HIDDEN), eo.dtype, storage_offset=0).data_ptr() for r in range(ws)]
    rank_ptrs = torch.tensor([ctypes.c_int64(p).value for p in ptrs], dtype=torch.int64, device=dev)
    out = torch.zeros(TOKENS_PER_RANK, HIDDEN, device=dev, dtype=eo.dtype)

    def timeit(fn, label):
        for _ in range(5):
            fn()
        torch.xpu.synchronize(); dist.barrier()
        ev0 = [torch.xpu.Event(enable_timing=True) for _ in range(ITERS)]
        ev1 = [torch.xpu.Event(enable_timing=True) for _ in range(ITERS)]
        for i in range(ITERS):
            ev0[i].record(); fn(); ev1[i].record()
        torch.xpu.synchronize()
        ts = sorted(ev0[i].elapsed_time(ev1[i]) for i in range(ITERS))
        med = ts[len(ts)//2]
        if rank == 0:
            print(f"[{label}] median={med:.3f} ms  min={ts[0]:.3f} max={ts[-1]:.3f}")

    timeit(lambda: torch.ops.symm_mem.ep_combine(eo, rank_ptrs, topk_idx, scatter_idx, tw, out, NUM_EXPERTS, rank, ws), "kernel_only")
    timeit(lambda: recv.barrier(), "barrier")
    def reduce():
        out.zero_()
        for i in range(ws):
            out.add_(my_buf[i])
    timeit(reduce, "reduce_py")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
