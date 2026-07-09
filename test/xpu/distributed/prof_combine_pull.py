"""Compare PUSH vs PULL (flashinfer gather-style) ep_combine kernels."""
import os
import ctypes
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from allgather_local_permute_fusion import compute_scatter_idx

torch.ops.load_library(
    os.path.join(os.path.dirname(__file__), "..", "csrc", "libep_combine.so"))


TOKENS_PER_RANK = 4096
HIDDEN = 7168
TOPK = 8
NUM_EXPERTS = 256
ITERS = 30


def main():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29655"
    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    torch.xpu.set_device(rank)
    dev = f"xpu:{rank}"
    gname = dist.group.WORLD.group_name

    ntok = TOKENS_PER_RANK * ws
    torch.manual_seed(42)
    topk_idx = torch.randint(0, NUM_EXPERTS, (ntok, TOPK), device=dev, dtype=torch.int32)
    scatter_idx, _ = compute_scatter_idx(topk_idx, num_experts=NUM_EXPERTS)
    torch.manual_seed(777)
    tw = torch.rand(ntok, TOPK, device=dev, dtype=torch.float32)
    tw = (tw / tw.sum(1, keepdim=True)).float()

    # expert_output is LOCAL (phase 1 reads it locally); no need for symmetric.
    eo = torch.randn(ntok * TOPK, HIDDEN, device=dev, dtype=torch.bfloat16)

    # Symmetric partial buffer [num_global_tokens, hidden] for the pull path:
    # phase 1 writes this rank's owner-side pre-aggregated rows here; phase 2
    # reads peers' partial buffers.
    partial = symm_mem.empty(ntok, HIDDEN, device=dev, dtype=torch.bfloat16)
    partial.zero_()
    phdl = symm_mem.rendezvous(partial, gname)
    partial_ptrs = [phdl.get_buffer(r, (ntok, HIDDEN), partial.dtype, storage_offset=0).data_ptr()
                    for r in range(ws)]
    partial_rank_ptrs = torch.tensor([ctypes.c_int64(p).value for p in partial_ptrs],
                                     dtype=torch.int64, device=dev)
    print(f"[{rank}] setup done", flush=True)

    # recv workspace for the push path.
    recv_size = ws * TOKENS_PER_RANK * HIDDEN * eo.element_size()
    recv = symm_mem.get_symm_mem_workspace(gname, min_size=recv_size)
    recv_ptrs = [recv.get_buffer(r, (ws, TOKENS_PER_RANK, HIDDEN), eo.dtype, storage_offset=0).data_ptr()
                 for r in range(ws)]
    recv_rank_ptrs = torch.tensor([ctypes.c_int64(p).value for p in recv_ptrs],
                                  dtype=torch.int64, device=dev)

    out_push = torch.zeros(TOKENS_PER_RANK, HIDDEN, device=dev, dtype=eo.dtype)
    out_pull = torch.zeros(TOKENS_PER_RANK, HIDDEN, device=dev, dtype=eo.dtype)

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
        med = ts[len(ts) // 2]
        print(f"[{rank}] [{label}] median={med:.3f} ms  min={ts[0]:.3f} max={ts[-1]:.3f} "
              f"p90={ts[int(len(ts)*0.9)]:.3f}")

    push = lambda: torch.ops.symm_mem.ep_combine(
        eo, recv_rank_ptrs, topk_idx, scatter_idx, tw, out_push, NUM_EXPERTS, rank, ws)
    phase1 = lambda: torch.ops.symm_mem.ep_combine_pull_partial(
        eo, partial, topk_idx, scatter_idx, tw, NUM_EXPERTS, rank, ws)
    gather = lambda: torch.ops.symm_mem.ep_combine_pull_gather(
        partial_rank_ptrs, topk_idx, out_pull, NUM_EXPERTS, rank, ws)

    def full_pull():
        phase1(); torch.xpu.synchronize(); dist.barrier(); gather()

    def check_pull():
        phase1(); torch.xpu.synchronize(); dist.barrier(); gather(); torch.xpu.synchronize()
        t0 = rank * TOKENS_PER_RANK
        sl = slice(t0, t0 + TOKENS_PER_RANK)
        si = scatter_idx[sl].long()
        w = tw[sl]
        gathered = eo[si.reshape(-1)].reshape(TOKENS_PER_RANK, TOPK, HIDDEN).float()
        ref = (w.unsqueeze(-1) * gathered).sum(1).to(eo.dtype)
        err = (ref.float() - out_pull.float()).abs().max().item()
        rel = err / (ref.float().abs().max().item() + 1e-6)
        print(f"[{rank}] pull correctness: max_abs_err={err:.4f} rel={rel:.5f}", flush=True)

    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("both", "push"):
        timeit(push, "push")
        dist.barrier()
    if mode in ("both", "pull"):
        check_pull()
        dist.barrier()
        timeit(phase1, "pull-phase1(local)")
        dist.barrier()
        timeit(gather, "pull-phase2(gather)")
        dist.barrier()
        timeit(full_pull, "pull-full(+barrier)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
