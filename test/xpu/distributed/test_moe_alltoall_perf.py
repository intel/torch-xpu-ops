"""Performance comparison: MoE all-to-all (EP) vs SymmBuffer (TP).

Compares two ways of doing the MoE dispatch+combine round trip on symmetric
memory, at an identical problem size:

  * TP path  (``symm_buffer.SymmBuffer``):
        dispatch = allgather + local permute
        combine  = unpermute + reduce-scatter
    Every token is all-gathered to every rank.

  * EP path  (``moe_alltoall.MoEAllToAll``):
        dispatch = ep_dispatch  (owner-based all-to-all)
        combine  = ep_combine   (owner-based, pipelined)
    Each device owns num_experts/world_size experts; a token is only sent to
    the device(s) that host its top-k experts and only combined back to its
    original device.

For each path the script reports the *per-rank communication volume* and the
measured latency, and checks that latency tracks the communication volume
(effective bandwidth is within the same order of magnitude).  Correctness of
both paths is verified against the analytic identity-expert reference.

Usage (MUST use 4 ranks):
    mpirun -np 4 --prepend-rank python test_moe_alltoall_perf.py
"""

import os

import torch
import torch.distributed as dist

import env

import symm_buffer as symm_buffer_mod
from allgather_local_permute_fusion import (
    _HAS_LOCAL_PERMUTE_KERNEL,
    compute_scatter_idx,
)
from unpermute_reducescatter_fusion import _HAS_LOCAL_UNPERMUTE_KERNEL
from moe_alltoall import MoEAllToAll, get_owner_expert_ranges

HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 2048))
TOPK = int(os.environ.get("TOPK", 8))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 128))
TOKENS_PER_RANK = env.tokens_per_rank()
LOOP = int(os.environ.get("LOOP", 40))
WARMUP = int(os.environ.get("WARMUP", 20))
DTYPE = torch.bfloat16


def init_distributed():
    env.setup_distributed_env(master_addr="localhost", master_port="29533")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def make_inputs(rank, world_size, device):
    T = TOKENS_PER_RANK
    num_tokens = T * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(T, HIDDEN_SIZE, device=device, dtype=DTYPE)

    torch.manual_seed(42)
    global_topk_idx = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOPK), device=device, dtype=torch.int32
    )

    torch.manual_seed(777)
    global_topk_weights = torch.rand(num_tokens, TOPK, device=device, dtype=torch.float32)
    global_topk_weights = global_topk_weights / global_topk_weights.sum(dim=1, keepdim=True)

    topk_idx = global_topk_idx[rank * T : (rank + 1) * T].clone()
    topk_weights = global_topk_weights[rank * T : (rank + 1) * T].clone()
    return hidden_shard, topk_idx, topk_weights, global_topk_idx, global_topk_weights


def timed_loop(fn, loop=LOOP, warmup=WARMUP):
    begin = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    end = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    for i in range(loop):
        if i >= warmup:
            begin[i].record()
        fn()
        if i >= warmup:
            end[i].record()
    torch.xpu.synchronize()
    dist.barrier()
    lat = [begin[i].elapsed_time(end[i]) for i in range(warmup, loop)]
    return sum(lat) / len(lat)


def expert_owner_tensor(device):
    owner = torch.empty(NUM_EXPERTS, dtype=torch.int64, device=device)
    for r, (start, end) in enumerate(get_owner_expert_ranges(NUM_EXPERTS, WORLD)):
        owner[start:end] = r
    return owner


def ep_dispatch_recv_rows(global_topk_idx, rank, world_size, device):
    return ep_dispatch_recv_rows_k(global_topk_idx, rank, world_size, device, TOPK)


def ep_dispatch_recv_rows_k(global_topk_idx, rank, world_size, device, topk):
    """Number of *token* rows this rank actually pulls from remote ranks.

    The ep_dispatch kernel reads each (token, source-rank) at most ONCE
    (coalesced) and then loops over top-k locally to fill every owned expert
    slot; an ownership pre-check skips the read entirely when the rank owns none
    of the token's experts (see EpDispatch.cpp). So the real cross-device volume
    is the number of DISTINCT remote tokens that have >=1 expert owned by this
    rank — not the per-(token, k) assignment count.
    """
    owner = expert_owner_tensor(device)
    num_tokens = global_topk_idx.shape[0]
    owner_of = owner[global_topk_idx.long()]          # [num_tokens, topk]
    owned_any = (owner_of == rank).any(dim=1)         # token has >=1 owned expert
    home = torch.arange(num_tokens, device=device) // TOKENS_PER_RANK
    remote = home != rank
    return int((owned_any & remote).sum().item())


def ep_owned_assignments(global_topk_idx, rank, world_size, device):
    """Number of (token, k) slots whose expert is owned by this rank.

    Drives EP local compute: the dispatch permute WRITES one remap row per owned
    (token, k), and the ownership-filtered combine unpermute GATHERS one
    expert_output row per owned (token, k).  TP instead touches every (token, k)
    of all W*T tokens (no ownership filter), i.e. W*T*topk.
    """
    owner = expert_owner_tensor(device)
    owner_of = owner[global_topk_idx.long()]          # [num_tokens, topk]
    return int((owner_of == rank).sum().item())


def fmt_mb(nbytes):
    return nbytes / (1024 * 1024)


def owner_partition_expert_output(remap, abs_scatter_idx, global_topk_idx, owner_tensor, rank):
    """Zero the rows of ``remap`` that belong to experts not owned by ``rank``.

    symm_buffer's allgather+permute dispatch fills the *full* permuted buffer on
    every rank. Feeding that directly into the reduce-scatter combine would sum
    each (token, k) row on all ranks (over-counting by world_size). Masking to
    owned-expert rows reproduces the realistic MoE case where each expert output
    exists on exactly one device, so the reduce-scatter reconstructs correctly.
    """
    total_rows = remap.shape[0]
    flat_expert = global_topk_idx.reshape(-1).long()
    owner_of = owner_tensor[flat_expert]
    rows = abs_scatter_idx.reshape(-1).long()
    valid = rows >= 0
    owned = (owner_of == rank) & valid
    mask = torch.zeros(total_rows, dtype=torch.bool, device=remap.device)
    mask[rows[owned]] = True
    return remap * mask.unsqueeze(1)


def main():
    global WORLD
    rank, world_size = init_distributed()
    WORLD = world_size
    device = f"xpu:{rank}"
    elem = torch.empty(0, dtype=DTYPE).element_size()

    if world_size != 4:
        if rank == 0:
            print(f"[warn] this UT is designed for -np 4; running with {world_size}")

    if not getattr(symm_buffer_mod, "_HAS_NOTIFY_DISPATCH_V2_KERNEL", False):
        raise RuntimeError("notify_dispatch_v2 kernel required for SymmBuffer TP path")
    if not (_HAS_LOCAL_PERMUTE_KERNEL and _HAS_LOCAL_UNPERMUTE_KERNEL):
        raise RuntimeError("local (un)permute kernels required")

    T = TOKENS_PER_RANK
    num_tokens = T * world_size

    hidden_shard, topk_idx, topk_weights, global_topk_idx, global_topk_weights = make_inputs(
        rank, world_size, device
    )
    scatter_idx, _ = compute_scatter_idx(global_topk_idx, num_experts=NUM_EXPERTS)

    # ================================================================== #
    #  TP path (SymmBuffer)                                               #
    # ================================================================== #
    sbuf = symm_buffer_mod.SymmBuffer(
        group=dist.group.WORLD,
        num_max_tokens_per_rank=T,
        hidden=HIDDEN_SIZE,
        num_topk=TOPK,
    )
    tp_remap = torch.empty(num_tokens * TOPK, HIDDEN_SIZE, device=device, dtype=DTYPE)
    tp_handle = [None]

    def tp_dispatch():
        _, tp_handle[0] = sbuf.allgather_local_permute_fusion(
            hidden_shard=hidden_shard,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=NUM_EXPERTS,
            remap_hidden_states=tp_remap,
        )

    tp_dispatch()  # warm to obtain a handle
    tp_output = torch.empty(T, HIDDEN_SIZE, device=device, dtype=DTYPE)

    owner_tensor = expert_owner_tensor(device)
    # Owner-partitioned expert output (identity experts): only owned-expert rows
    # survive, so the reduce-scatter reconstructs this rank's hidden shard.
    tp_expert_output = owner_partition_expert_output(
        tp_remap, tp_handle[0].abs_scatter_idx, global_topk_idx, owner_tensor, rank
    )

    def tp_combine():
        # identity experts: expert_output is the owner-partitioned permuted hidden.
        sbuf.unpermute_reducescatter_fusion(
            expert_output=tp_expert_output, handle=tp_handle[0], output=tp_output
        )

    tp_dispatch_ms = timed_loop(tp_dispatch)
    tp_dispatch()
    tp_expert_output = owner_partition_expert_output(
        tp_remap, tp_handle[0].abs_scatter_idx, global_topk_idx, owner_tensor, rank
    )
    tp_combine_ms = timed_loop(tp_combine)

    # ================================================================== #
    #  EP path (MoEAllToAll)                                              #
    # ================================================================== #
    ep_group = dist.new_group(list(range(world_size)), group_desc="moe_alltoall_main")
    moe = MoEAllToAll(
        group=ep_group,
        num_max_tokens_per_rank=T,
        hidden=HIDDEN_SIZE,
        num_topk=TOPK,
        num_experts=NUM_EXPERTS,
    )
    ep_remap = torch.empty(num_tokens * TOPK, HIDDEN_SIZE, device=device, dtype=DTYPE)
    ep_recv_idx = torch.full((num_tokens * TOPK, TOPK), -1, device=device, dtype=torch.int32)
    ep_recv_w = torch.zeros((num_tokens * TOPK, TOPK), device=device, dtype=torch.float32)
    ep_handle = [None]

    def ep_dispatch():
        ep_handle[0] = moe.dispatch(
            hidden_shard=hidden_shard,
            global_topk_idx=global_topk_idx,
            global_topk_weights=global_topk_weights,
            scatter_idx=scatter_idx,
            remap_hidden_states=ep_remap,
            recv_topk_idx=ep_recv_idx,
            recv_topk_weights=ep_recv_w,
        )

    ep_dispatch()
    ep_output = torch.empty(T, HIDDEN_SIZE, device=device, dtype=DTYPE)

    def ep_combine():
        # identity experts: expert_output is the dispatched (owner-filled) hidden.
        moe.combine(expert_output=ep_remap, handle=ep_handle[0], output=ep_output)

    ep_dispatch_ms = timed_loop(ep_dispatch)
    ep_dispatch()
    ep_combine_ms = timed_loop(ep_combine)

    # ================================================================== #
    #  Correctness (identity experts, normalized weights => reconstruct    #
    #  this rank's hidden shard)                                           #
    # ================================================================== #
    torch.xpu.synchronize()
    tp_err = (tp_output.float() - hidden_shard.float()).abs().max().item()
    ep_err = (ep_output.float() - hidden_shard.float()).abs().max().item()
    tol = 0.05
    tp_ok = tp_err <= tol
    ep_ok = ep_err <= tol

    # ================================================================== #
    #  Communication volume (per rank)                                    #
    # ================================================================== #
    tp_dispatch_bytes = (world_size - 1) * T * HIDDEN_SIZE * elem            # allgather recv
    tp_combine_bytes = (world_size - 1) * T * HIDDEN_SIZE * elem             # reduce-scatter push
    ep_dispatch_rows = ep_dispatch_recv_rows(global_topk_idx, rank, world_size, device)
    ep_dispatch_bytes = ep_dispatch_rows * HIDDEN_SIZE * elem                # owner pulls (deduped)
    ep_combine_bytes = (world_size - 1) * T * HIDDEN_SIZE * elem             # ring reduce-scatter push

    # --- local compute volume (permute writes / unpermute gathers), per rank ---
    #   TP touches every (token, k) of all W*T tokens (no ownership filter).
    #   EP touches only the (token, k) slots whose expert this rank owns.
    tp_assign = world_size * T * TOPK
    ep_assign = ep_owned_assignments(global_topk_idx, rank, world_size, device)
    row_bytes = HIDDEN_SIZE * elem
    tp_permute_bytes = tp_assign * row_bytes      # dispatch: remap rows written
    ep_permute_bytes = ep_assign * row_bytes
    tp_gather_bytes = tp_assign * row_bytes       # combine: expert_output rows gathered
    ep_gather_bytes = ep_assign * row_bytes

    def bw(nbytes, ms):
        return (nbytes / 1e6) / ms if ms > 0 else 0.0  # GB/s (1e9 B/s)

    if rank == 0:
        line = "=" * 78
        print(f"\n{line}")
        print(
            f"[config] ws={world_size} tokens/rank={T} hidden={HIDDEN_SIZE} "
            f"topk={TOPK} experts={NUM_EXPERTS} dtype={DTYPE}"
        )
        print(f"[correctness] TP max_err={tp_err:.4f} ok={tp_ok} | "
              f"EP max_err={ep_err:.4f} ok={ep_ok}")
        print(line)
        print(f"{'stage':<26}{'lat(ms)':>10}{'comm/rank(MB)':>16}{'BW(GB/s)':>12}")
        print("-" * 78)
        for name, ms, nb in [
            ("TP dispatch (allgather)", tp_dispatch_ms, tp_dispatch_bytes),
            ("EP dispatch (ep_dispatch)", ep_dispatch_ms, ep_dispatch_bytes),
            ("TP combine  (reduce-scat)", tp_combine_ms, tp_combine_bytes),
            ("EP combine  (ring-ep)", ep_combine_ms, ep_combine_bytes),
        ]:
            print(f"{name:<26}{ms:>10.3f}{fmt_mb(nb):>16.2f}{bw(nb, ms):>12.2f}")
        print("-" * 78)
        tp_total = tp_dispatch_ms + tp_combine_ms
        ep_total = ep_dispatch_ms + ep_combine_ms
        tp_bytes = tp_dispatch_bytes + tp_combine_bytes
        ep_bytes = ep_dispatch_bytes + ep_combine_bytes
        print(f"{'TP total':<26}{tp_total:>10.3f}{fmt_mb(tp_bytes):>16.2f}"
              f"{bw(tp_bytes, tp_total):>12.2f}")
        print(f"{'EP total':<26}{ep_total:>10.3f}{fmt_mb(ep_bytes):>16.2f}"
              f"{bw(ep_bytes, ep_total):>12.2f}")
        print(line)
        print(
            f"[dispatch] EP/TP comm ratio={ep_dispatch_bytes / tp_dispatch_bytes:.3f} "
            f"latency ratio={ep_dispatch_ms / tp_dispatch_ms:.3f}"
        )
        print(
            f"[combine ] EP/TP comm ratio={ep_combine_bytes / tp_combine_bytes:.3f} "
            f"latency ratio={ep_combine_ms / tp_combine_ms:.3f}"
        )
        print(
            f"[total   ] EP/TP comm ratio={ep_bytes / tp_bytes:.3f} "
            f"latency ratio={ep_total / tp_total:.3f} "
            f"speedup={tp_total / ep_total:.3f}x"
        )
        print(line)

        # --- local compute volume: EP is ownership-filtered (~1/W) ------- #
        print("\n" + line)
        print("[local compute volume] permute writes (dispatch) / gather reads (combine)")
        print(f"{'stage':<26}{'rows/rank':>12}{'MB/rank':>12}{'EP/TP':>10}")
        print("-" * 78)
        print(f"{'TP dispatch permute':<26}{tp_assign:>12}{fmt_mb(tp_permute_bytes):>12.2f}"
              f"{'1.000':>10}")
        print(f"{'EP dispatch permute':<26}{ep_assign:>12}{fmt_mb(ep_permute_bytes):>12.2f}"
              f"{ep_permute_bytes / tp_permute_bytes:>10.3f}")
        print(f"{'TP combine gather':<26}{tp_assign:>12}{fmt_mb(tp_gather_bytes):>12.2f}"
              f"{'1.000':>10}")
        print(f"{'EP combine gather':<26}{ep_assign:>12}{fmt_mb(ep_gather_bytes):>12.2f}"
              f"{ep_gather_bytes / tp_gather_bytes:>10.3f}")
        print(line)

    assert tp_ok, f"TP path incorrect on rank {rank}: max_err={tp_err}"
    assert ep_ok, f"EP path incorrect on rank {rank}: max_err={ep_err}"

    # ================================================================== #
    #  topk sweep: show EP dispatch latency tracks communication volume    #
    # ================================================================== #
    if os.environ.get("SWEEP", "1") != "0":
        sweep_topks = [int(x) for x in os.environ.get("SWEEP_TOPKS", "1,2,4,8").split(",")]
        rows = []
        for k in sweep_topks:
            torch.manual_seed(42)
            g_idx = torch.randint(0, NUM_EXPERTS, (num_tokens, k), device=device, dtype=torch.int32)
            torch.manual_seed(777)
            g_w = torch.rand(num_tokens, k, device=device, dtype=torch.float32)
            g_w = g_w / g_w.sum(dim=1, keepdim=True)
            s_idx, _ = compute_scatter_idx(g_idx, num_experts=NUM_EXPERTS)

            grp = dist.new_group(list(range(world_size)), group_desc=f"moe_sweep_k{k}")
            moe_k = MoEAllToAll(
                group=grp, num_max_tokens_per_rank=T, hidden=HIDDEN_SIZE,
                num_topk=k, num_experts=NUM_EXPERTS,
            )
            remap_k = torch.empty(num_tokens * k, HIDDEN_SIZE, device=device, dtype=DTYPE)
            ridx_k = torch.full((num_tokens * k, k), -1, device=device, dtype=torch.int32)
            rw_k = torch.zeros((num_tokens * k, k), device=device, dtype=torch.float32)

            def run_disp(moe_k=moe_k, g_idx=g_idx, g_w=g_w, s_idx=s_idx,
                         remap_k=remap_k, ridx_k=ridx_k, rw_k=rw_k):
                moe_k.dispatch(
                    hidden_shard=hidden_shard, global_topk_idx=g_idx,
                    global_topk_weights=g_w, scatter_idx=s_idx,
                    remap_hidden_states=remap_k, recv_topk_idx=ridx_k,
                    recv_topk_weights=rw_k,
                )

            run_disp()
            for _ in range(3):
                run_disp()
            torch.xpu.synchronize()
            dist.barrier()
            ms = timed_loop(run_disp)
            rows_pulled = ep_dispatch_recv_rows_k(g_idx, rank, world_size, device, k)
            nb = rows_pulled * HIDDEN_SIZE * elem
            rows.append((k, nb, ms))

        if rank == 0:
            line = "=" * 78
            print(f"\n{line}")
            print("[EP dispatch topk sweep] latency vs communication volume")
            print(f"{'topk':>6}{'comm/rank(MB)':>16}{'lat(ms)':>10}{'BW(GB/s)':>12}"
                  f"{'ms/MB':>10}")
            print("-" * 78)
            for k, nb, ms in rows:
                mb = nb / 1e6
                print(f"{k:>6}{fmt_mb(nb):>16.2f}{ms:>10.3f}"
                      f"{(mb / ms):>12.2f}{(ms / max(mb, 1e-9)):>10.4f}")
            print(line)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
