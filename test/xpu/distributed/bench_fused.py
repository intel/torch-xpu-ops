"""
Benchmark: fused (in-kernel barrier) vs original (4 kernel launches).

Measures both approaches across a range of tokens_per_rank to find the
crossover point.

Usage:
    cd test/xpu/csrc && python build.py   # build all kernels
    cd test/xpu/distributed
    mpirun -n 4 python bench_fused.py
    mpirun -n 4 python bench_fused.py 128 256 512  # specific sizes
"""

import ctypes
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from allgather_local_permute_fusion import (
    allgather_local_permute_fusion,
    build_allgather_rank_buffers_ptr,
    compute_scatter_idx,
)

HIDDEN_SIZE = 2048
TOPK = 8
NUM_EXPERTS = 128
LOOP = 40
WARMUP = 20

# Load fused kernel
_FUSED_LIB = os.path.join(os.path.dirname(__file__), "..", "csrc", "liballgather_permute_fused.so")
_HAS_FUSED = False
if os.path.exists(_FUSED_LIB):
    try:
        torch.ops.load_library(_FUSED_LIB)
        _HAS_FUSED = hasattr(torch.ops.symm_mem, "allgather_permute_fused")
    except Exception as e:
        print(f"Warning: failed to load fused kernel: {e}")

# Also load the original kernel (needed by allgather_local_permute_fusion)
_ORIG_LIB = os.path.join(os.path.dirname(__file__), "..", "csrc", "liblocal_permute_copy.so")
if os.path.exists(_ORIG_LIB):
    try:
        torch.ops.load_library(_ORIG_LIB)
    except Exception:
        pass


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29535"
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size


def build_fused_buffers(hidden_shard, group, group_name):
    """Build rank_buffers_ptr and sync_bufs_ptr.

    Data buffers use the standard workspace.
    Sync buffers placed after data region in workspace.
    MUST be called with fresh workspace (not stale pointers).

    Returns (rank_buffers_ptr, sync_bufs_ptr, sync_local_buf) tensors.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    num_tokens_per_rank, hidden_size = hidden_shard.shape

    data_bytes = num_tokens_per_rank * hidden_size * hidden_shard.element_size() * world_size
    sync_slots = 2 * world_size
    sync_bytes = sync_slots * 4
    data_bytes_aligned = ((data_bytes + 15) // 16) * 16
    total_size = data_bytes_aligned + sync_bytes

    workspace = symm_mem.get_symm_mem_workspace(group_name, min_size=total_size)

    rank_ptr_list = []
    sync_ptr_list = []
    for r in range(world_size):
        data_offset = r * num_tokens_per_rank * hidden_size
        buf = workspace.get_buffer(
            r, (num_tokens_per_rank, hidden_size),
            hidden_shard.dtype, storage_offset=data_offset,
        )
        rank_ptr_list.append(buf.data_ptr())
        sync_offset_int32 = data_bytes_aligned // 4
        sync_buf = workspace.get_buffer(
            r, (sync_slots,), torch.int32, storage_offset=sync_offset_int32,
        )
        sync_ptr_list.append(sync_buf.data_ptr())

    my_sync = workspace.get_buffer(
        rank, (sync_slots,), torch.int32,
        storage_offset=data_bytes_aligned // 4,
    )

    device = hidden_shard.device
    rank_signed = [ctypes.c_int64(p).value for p in rank_ptr_list]
    sync_signed = [ctypes.c_int64(p).value for p in sync_ptr_list]
    rank_bufs = torch.tensor(rank_signed, dtype=torch.int64).to(device)
    sync_bufs = torch.tensor(sync_signed, dtype=torch.int64).to(device)
    return rank_bufs, sync_bufs, my_sync


def bench_original(rank, world_size, tokens_per_rank):
    """Benchmark the original 4-launch approach."""
    device = f"xpu:{rank}"
    group = dist.group.WORLD
    num_tokens = tokens_per_rank * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    torch.manual_seed(42)
    topk_idx = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOPK), device=device, dtype=torch.int64
    )
    scatter_idx, _ = compute_scatter_idx(topk_idx, num_experts=NUM_EXPERTS)
    rank_buffers = build_allgather_rank_buffers_ptr(hidden_shard, group=group)

    events_b = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    events_e = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    # Warmup
    for _ in range(WARMUP):
        remap = torch.empty(
            (num_tokens * TOPK, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
        )
        allgather_local_permute_fusion(
            hidden_shard, topk_idx, scatter_idx, remap,
            group=group, rank_buffers_ptr=rank_buffers,
        )
    torch.xpu.synchronize()
    dist.barrier()

    remap = torch.empty(
        (num_tokens * TOPK, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    for i in range(LOOP):
        if i >= WARMUP:
            events_b[i].record()
        allgather_local_permute_fusion(
            hidden_shard, topk_idx, scatter_idx, remap,
            group=group, rank_buffers_ptr=rank_buffers,
        )
        if i >= WARMUP:
            events_e[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    lats = [events_b[i].elapsed_time(events_e[i]) for i in range(WARMUP, LOOP)]
    # Return remap for correctness check
    return lats, remap


def bench_fused(rank, world_size, tokens_per_rank, generation_start=0):
    """Benchmark the fused 1-launch approach."""
    device = f"xpu:{rank}"
    group = dist.group.WORLD
    group_name = group.group_name
    num_tokens = tokens_per_rank * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        tokens_per_rank, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    torch.manual_seed(42)
    topk_idx = torch.randint(
        0, NUM_EXPERTS, (num_tokens, TOPK), device=device, dtype=torch.int64
    )
    scatter_idx, _ = compute_scatter_idx(topk_idx, num_experts=NUM_EXPERTS)

    # Build fused-specific buffers (get fresh pointers)
    fused_rank_bufs, sync_bufs, my_sync = build_fused_buffers(
        hidden_shard, group, group_name
    )
    # Zero sync and create grid state
    my_sync.zero_()
    torch.xpu.synchronize()
    dist.barrier()
    grid_state = torch.zeros(4, dtype=torch.int32, device=device)

    events_b = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    events_e = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    gen = generation_start

    # Warmup
    for _ in range(WARMUP):
        remap = torch.empty(
            (num_tokens * TOPK, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
        )
        torch.ops.symm_mem.allgather_permute_fused(
            hidden_shard, fused_rank_bufs, scatter_idx, remap,
            sync_bufs, grid_state, rank, world_size, gen,
        )
        gen += 1
    torch.xpu.synchronize()
    dist.barrier()

    remap = torch.empty(
        (num_tokens * TOPK, HIDDEN_SIZE), device=device, dtype=torch.bfloat16
    )
    for i in range(LOOP):
        if i >= WARMUP:
            events_b[i].record()
        torch.ops.symm_mem.allgather_permute_fused(
            hidden_shard, fused_rank_bufs, scatter_idx, remap,
            sync_bufs, grid_state, rank, world_size, gen,
        )
        gen += 1
        if i >= WARMUP:
            events_e[i].record()
    torch.xpu.synchronize()
    dist.barrier()

    lats = [events_b[i].elapsed_time(events_e[i]) for i in range(WARMUP, LOOP)]
    return lats, remap


def main():
    rank, world_size = init_distributed()

    sizes = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [64, 128, 256, 512, 1024]

    if rank == 0:
        print(f"\n{'='*72}")
        print(f"  Fused vs Original Allgather+Permute Benchmark")
        print(f"  hidden={HIDDEN_SIZE}, topk={TOPK}, experts={NUM_EXPERTS}")
        print(f"  world_size={world_size}, fused_kernel={'available' if _HAS_FUSED else 'NOT FOUND'}")
        print(f"{'='*72}")

    if not _HAS_FUSED:
        if rank == 0:
            print("ERROR: fused kernel not built. Run: cd test/xpu/csrc && python build.py")
        dist.destroy_process_group()
        return

    # Pre-allocate symmetric memory workspace for the largest data size to
    # avoid workspace reallocation when token count changes mid-run, which
    # triggers DEVICE_LOST on Intel GPUs.  The fused kernel's build_fused_buffers
    # may request a slightly larger workspace (adds sync_bytes), but that small
    # reallocation is benign — it's the large data-size jumps that crash.
    max_tpr = max(sizes)
    max_data_bytes = max_tpr * HIDDEN_SIZE * 2 * world_size  # bf16 = 2 bytes
    group = dist.group.WORLD
    symm_mem.get_symm_mem_workspace(group.group_name, min_size=max_data_bytes)
    torch.xpu.synchronize()
    dist.barrier()

    gen = 0
    for tpr in sizes:
        # Original
        if rank == 0: print(f"\n  [tokens_per_rank={tpr}] Running original...", flush=True)
        lats_orig, remap_orig = bench_original(rank, world_size, tpr)
        if rank == 0: print(f"  [tokens_per_rank={tpr}] Running fused...", flush=True)

        # Fused (build_fused_buffers gets fresh pointers)
        lats_fused, remap_fused = bench_fused(rank, world_size, tpr, generation_start=gen)
        gen += WARMUP + LOOP

        # Correctness check — use bit-level comparison so NaN == NaN.
        # workspace.barrier() on Intel XPU may occasionally race, producing
        # a small number of stale-read diffs; report count instead of bare bool.
        if rank == 0:
            orig_bits = remap_orig.view(torch.int16)
            fused_bits = remap_fused.view(torch.int16)
            n_diff = int((orig_bits != fused_bits).sum())
            if n_diff == 0:
                correct_tag = "correct=True"
            else:
                total = orig_bits.numel()
                correct_tag = f"correct=False ({n_diff}/{total} diffs)"

            avg_o = sum(lats_orig) / len(lats_orig)
            med_o = sorted(lats_orig)[len(lats_orig) // 2]
            min_o = min(lats_orig)

            avg_f = sum(lats_fused) / len(lats_fused)
            med_f = sorted(lats_fused)[len(lats_fused) // 2]
            min_f = min(lats_fused)

            speedup = avg_o / avg_f if avg_f > 0 else 0
            delta = avg_o - avg_f

            VEC_SIZE = 8
            hidden_vecs = HIDDEN_SIZE // VEC_SIZE
            total_wi = world_size * tpr * hidden_vecs
            orig_wgs = (total_wi + 255) // 256

            print(f"\n  tokens_per_rank={tpr}  (orig_WGs={orig_wgs})")
            print(f"    Original:  avg={avg_o:.3f}  med={med_o:.3f}  min={min_o:.3f} ms")
            print(f"    Fused:     avg={avg_f:.3f}  med={med_f:.3f}  min={min_f:.3f} ms")
            print(f"    Δ={delta:+.3f} ms  speedup={speedup:.2f}x  {correct_tag}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
