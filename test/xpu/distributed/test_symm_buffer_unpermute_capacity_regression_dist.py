"""Regression test for the ring unpermute + reduce-scatter staleness bug.

Bug: SymmBuffer.unpermute_reducescatter_fusion with FUSION_RING=1 drove its
cross-rank peer write with an LSC L1WB_L3WB (writeback) store. The system-scope
release fence ordered that store but did NOT reliably drain it to the coherence
point before the downstream peer's acquire-load read it on the next ring step, so
the peer folded a STALE acc block into the reduce-scatter and produced garbage
combine outputs.

Crucially, this reproduced whenever the LIVE token count differed from the
num_max_tokens_per_rank the SymmBuffer was created with -- NOT only for "small"
token counts. This is exactly the vLLM pattern: the buffer is sized once to the
(large) warmup batch, then every real prefill/decode step drives it with fewer
tokens. This test therefore creates ONE buffer at a large capacity and exercises
several live token counts strictly BELOW that capacity (plus one AT capacity as a
control), asserting the FUSION_RING=1 (ring) combine output matches the
FUSION_RING=0 (staged, known-good) path.

Usage:
    mpirun -n 4 python test_symm_buffer_unpermute_capacity_regression_dist.py
"""

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist

import symm_buffer as symm_buffer_mod
from symm_buffer import SymmBuffer

HIDDEN_SIZE = 1024
TOPK = 8
NUM_EXPERTS = 128
# Buffer capacity. The bug is triggered by live token counts BELOW this value.
NUM_MAX_TOKENS_PER_RANK = 2048
# Live per-rank token counts to exercise. All but the last are < capacity (the
# regression window); the last equals capacity as a passing control.
LIVE_TOKENS_PER_RANK = [1, 5, 64, 300, 2000, NUM_MAX_TOKENS_PER_RANK]


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29538"
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


@contextmanager
def force_fusion_ring(enabled):
    previous = symm_buffer_mod._FUSION_RING
    symm_buffer_mod._FUSION_RING = enabled
    try:
        yield
    finally:
        symm_buffer_mod._FUSION_RING = previous


def make_symm_buffer(enabled, group):
    with force_fusion_ring(enabled):
        sbuf = SymmBuffer(
            group=group,
            num_max_tokens_per_rank=NUM_MAX_TOKENS_PER_RANK,
            hidden=HIDDEN_SIZE,
            num_topk=TOPK,
        )
    if enabled and not getattr(sbuf, "_fusion_ring", False):
        raise RuntimeError(
            "FUSION_RING=1 requested but the ring backend is unavailable; "
            "cannot run this regression test."
        )
    return sbuf


def make_inputs(rank, world_size, ntok, step, device):
    num_tokens = ntok * world_size
    torch.manual_seed(1234 + rank + step)
    hidden_shard = torch.randn(ntok, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    torch.manual_seed(42 + step)
    global_topk_idx = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOPK), device=device, dtype=torch.int32
    )
    topk_idx = global_topk_idx[rank * ntok : (rank + 1) * ntok].clone()

    torch.manual_seed(777 + step)
    global_topk_weights = torch.rand(num_tokens, TOPK, device=device, dtype=torch.float32)
    global_topk_weights = global_topk_weights / global_topk_weights.sum(dim=1, keepdim=True)
    topk_weights = global_topk_weights[rank * ntok : (rank + 1) * ntok].clone()
    return hidden_shard, topk_idx, topk_weights


def build_consistent_expert_output(canonical_src, abs_scatter_idx):
    """Scatter the SAME per-(token, top-k) canonical data through a path's own
    abs_scatter_idx. Each fusion path assigns a different (but valid) within-expert
    row ordering, so this guarantees each path reads back identical logical data
    during combine -- exactly what a real expert stage would produce.
    """
    eo = torch.zeros_like(canonical_src)
    flat = abs_scatter_idx.reshape(-1)
    valid = flat >= 0
    eo[flat[valid].long()] = canonical_src[valid]
    return eo


def run_cycle(sbuf, hidden_shard, topk_idx, topk_weights, canonical_src, ntok, ws, device):
    remap = torch.empty(
        (ntok * ws * TOPK, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    _, handle = sbuf.allgather_local_permute_fusion(
        hidden_shard=hidden_shard,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=NUM_EXPERTS,
        remap_hidden_states=remap,
    )
    expert_output = build_consistent_expert_output(canonical_src, handle.abs_scatter_idx)
    output = torch.empty(ntok, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    sbuf.unpermute_reducescatter_fusion(
        expert_output=expert_output, handle=handle, output=output
    )
    return output


def check():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    if not getattr(symm_buffer_mod, "_HAS_NOTIFY_DISPATCH_V2_KERNEL", False):
        raise RuntimeError("notify_dispatch_v2 kernel required for this test")

    # Two independent buffers, each created ONCE at the large capacity.
    sbuf_ref = make_symm_buffer(False, group)   # FUSION_RING=0 (staged, known-good)
    sbuf_ring = make_symm_buffer(True, group)   # FUSION_RING=1 (ring, under test)

    atol = 0.025 * world_size + 0.02
    failures = []
    for step, ntok in enumerate(LIVE_TOKENS_PER_RANK):
        hidden_shard, topk_idx, topk_weights = make_inputs(
            rank, world_size, ntok, step, device
        )
        torch.manual_seed(9000 + step)
        canonical_src = torch.randn(
            ntok * world_size * TOPK, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
        )

        out_ref = run_cycle(
            sbuf_ref, hidden_shard, topk_idx, topk_weights, canonical_src, ntok, world_size, device
        )
        out_ring = run_cycle(
            sbuf_ring, hidden_shard, topk_idx, topk_weights, canonical_src, ntok, world_size, device
        )

        max_diff = (out_ref.float() - out_ring.float()).abs().max().item()
        ok = torch.allclose(out_ref, out_ring, atol=atol, rtol=1e-2)
        below = "below" if ntok < NUM_MAX_TOKENS_PER_RANK else "at"
        print(
            f"[Rank {rank}] live_tokens_per_rank={ntok:5d} ({below} capacity "
            f"{NUM_MAX_TOKENS_PER_RANK}) ring-vs-staged match={ok} max_diff={max_diff:.5f}",
            flush=True,
        )
        if not ok:
            failures.append((ntok, max_diff))
        dist.barrier()

    assert not failures, (
        f"[Rank {rank}] ring unpermute_reducescatter_fusion diverged from the staged "
        f"path at live token counts {[f'{n}(max_diff={d:.4f})' for n, d in failures]} "
        f"(atol={atol}). This is the writeback-store staleness regression: the peer "
        f"write in ring_reduce_scatter_unpermute must use a coherent (plain) store."
    )
    if rank == 0:
        print("[Summary] ring unpermute_reducescatter matches staged path for all "
              "live token counts below and at buffer capacity. PASSED")

    dist.destroy_process_group()


if __name__ == "__main__":
    check()
