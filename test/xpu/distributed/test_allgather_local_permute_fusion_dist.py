"""
Accuracy + performance check for allgather_local_permute_fusion (distributed style)

Usage:
    mpirun -n 2 python test_allgather_local_permute_fusion_dist.py
"""
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist

from allgather_local_permute_fusion import allgather_local_permute_fusion, allgather_with_symm_mem


TOKENS_PER_RANK = 2048
HIDDEN_SIZE = 2048
TOPK = 8
LOOP = 10
ENABLE_PROFILE = False

def init_distributed():
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29513'
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.xpu.set_device(rank)
    return rank, world_size

def check_allgather_local_permute_fusion():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    num_tokens_per_rank = TOKENS_PER_RANK
    hidden_size = HIDDEN_SIZE
    topk = TOPK
    num_tokens = num_tokens_per_rank * world_size

    # Each rank: unique hidden_shard
    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(num_tokens_per_rank, hidden_size, device=device, dtype=torch.bfloat16)

    # All ranks: same topk_idx
    torch.manual_seed(42)
    topk_idx = torch.randint(0, world_size, (num_tokens, topk), device=device, dtype=torch.int64)

    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    if ENABLE_PROFILE:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()

    with prof:
        # Warm up fused path
        for _ in range(LOOP):
            remap_hidden_states = torch.empty((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)
            output_fused = allgather_local_permute_fusion(
                hidden_shard,
                topk_idx,
                remap_hidden_states=remap_hidden_states,
                group=group,
            )
        torch.xpu.synchronize()
        dist.barrier()

        remap_hidden_states = torch.empty((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)
        # Timed fused path
        for i in range(LOOP):
            begin_events[i].record()
            output_fused = allgather_local_permute_fusion(
                hidden_shard,
                topk_idx,
                remap_hidden_states=remap_hidden_states,
                group=group,
            )
            end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]

    if ENABLE_PROFILE:
        prof.export_chrome_trace(f"./profile_allgather_local_permute_fusion_rank{rank}.json")

    print(f"[Fusion time in rank {rank}] {latencies} ms")

    # Accuracy check: run one fresh pass and compare against reference
    remap_check = torch.zeros((num_tokens * topk, hidden_size), device=device, dtype=hidden_shard.dtype)
    allgather_local_permute_fusion(
        hidden_shard, topk_idx, remap_hidden_states=remap_check, group=group,
    )
    gathered = [torch.empty_like(hidden_shard) for _ in range(world_size)]
    dist.all_gather(gathered, hidden_shard, group=group)
    all_hidden = torch.stack(gathered, dim=0)  # [world_size, tokens_per_rank, hidden_size]
    ref = torch.zeros_like(remap_check)
    for src_rank in range(world_size):
        for i in range(num_tokens_per_rank):
            global_idx = src_rank * num_tokens_per_rank + i
            for k in range(topk):
                dst = global_idx * topk + k
                ref[dst].copy_(all_hidden[src_rank, i])
    assert torch.equal(remap_check, ref), f"allgather_local_permute_fusion mismatch in rank {rank}"

    if rank == 0:
        avg_fused = sum(latencies) / len(latencies)
        print(f"[Summary] avg_fused={avg_fused:.3f} ms")

    dist.destroy_process_group()

def check_allgather_with_symm_mem():
    rank, world_size = init_distributed()
    device = f"xpu:{rank}"
    group = dist.group.WORLD

    numel_per_rank = TOKENS_PER_RANK * HIDDEN_SIZE
    total_numel = numel_per_rank * world_size

    # Each rank: unique input
    torch.manual_seed(1234 + rank)
    input_shard = torch.randn(numel_per_rank, device=device, dtype=torch.bfloat16)

    print(f"[Allgather input size per rank] {input_shard.numel() * input_shard.element_size() / 1024 / 1024:.2f} MB")

    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(LOOP)]

    if ENABLE_PROFILE:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()

    with prof:
        # Warm up
        for _ in range(LOOP):
            output = torch.empty(total_numel, device=device, dtype=input_shard.dtype)
            allgather_with_symm_mem(
                input_shard,
                output_tensor=output,
                group=group,
            )
        torch.xpu.synchronize()
        dist.barrier()

        output = torch.empty(total_numel, device=device, dtype=input_shard.dtype)
        # Timed path
        for i in range(LOOP):
            begin_events[i].record()
            allgather_with_symm_mem(
                input_shard,
                output_tensor=output,
                group=group,
            )
            end_events[i].record()
        torch.xpu.synchronize()
        dist.barrier()

    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]

    if ENABLE_PROFILE:
        prof.export_chrome_trace(f"./profile_allgather_with_symm_mem_rank{rank}.json")

    print(f"[Allgather time in rank {rank}] {latencies} ms")

    # Accuracy check: gather all input_shard from all ranks and compare
    gathered = [torch.empty_like(input_shard) for _ in range(world_size)]
    dist.all_gather(gathered, input_shard, group=group)
    ref = torch.cat(gathered, dim=0)
    assert torch.allclose(output, ref, atol=1e-3), f"Allgather mismatch in rank {rank}"

    if rank == 0:
        avg = sum(latencies) / len(latencies)
        print(f"[Summary] allgather avg={avg:.3f} ms")

    dist.destroy_process_group()

if __name__ == "__main__":
    check_allgather_local_permute_fusion()
    #check_allgather_with_symm_mem()
