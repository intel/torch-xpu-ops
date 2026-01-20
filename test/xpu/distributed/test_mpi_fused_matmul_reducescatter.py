import itertools
import os
from contextlib import nullcontext
from unittest import skip, skipIf

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import (
    _fused_matmul_reduce_scatter_fallback,
    _fused_matmul_reduce_scatter,
    _test_mode,
    enable_symm_mem_for_group,
    restride_A_for_fused_matmul_reduce_scatter,
    restride_A_shard_for_fused_all_gather_matmul,
)
import os
import torch.multiprocessing as mp

BATCH = 1
M = 8192
N = 4096
K = 7168
Loop = 10
enable_profile = False

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29803'
dist.init_process_group(backend='xccl')

def test_matmul_reducescatter(rank, world_size):
    torch.xpu.set_device(rank)

    group = dist.group.WORLD

    torch.manual_seed(42 + rank)
    A = torch.rand(M, K, device="xpu")
    B = torch.rand(K, N, device="xpu")
    scatter_dim = 0

    begin_events_ref = [
        torch.xpu.Event(enable_timing=True) for _ in range(Loop)
    ]
    end_events_ref = [torch.xpu.Event(enable_timing=True) for _ in range(Loop)]

    begin_events = [
        torch.xpu.Event(enable_timing=True) for _ in range(Loop)
    ]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(Loop)]

    if enable_profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
    else:
        prof = nullcontext()

    with prof:
        # warm up fallback aten ops
        for i in range(Loop):
            output_0 = _fused_matmul_reduce_scatter_fallback(
                A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
            )
        torch.xpu.synchronize()
        for i in range(Loop):
            begin_events_ref[i].record()
            output_0 = _fused_matmul_reduce_scatter_fallback(
                A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
            )
            end_events_ref[i].record()
        torch.xpu.synchronize()

        # warm up symmetric ops
        for i in range(Loop):
            output_1 = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
            )
        torch.xpu.synchronize()
        for i in range(Loop):
            begin_events[i].record()
            output_1 = torch.ops.symm_mem.fused_matmul_reduce_scatter(
                A, B, "avg", scatter_dim=scatter_dim, group_name=group.group_name
            )
            end_events[i].record()
        torch.xpu.synchronize()

    latencies_ref = [b.elapsed_time(e) for b, e in zip(begin_events_ref, end_events_ref)]
    latencies = [b.elapsed_time(e) for b, e in zip(begin_events, end_events)]

    if enable_profile:
        prof.export_chrome_trace("./profile_kineto_trace_" + str(rank) + ".json")

    assert torch.allclose(output_0, output_1)
    assert output_0.stride() == output_1.stride()

    dist.destroy_process_group()
    print(f"[Fallback time in rank {rank}] {latencies_ref} ms")
    print(f"[Symm ops time in rank {rank}] {latencies} ms")

rank = dist.get_rank()
world_size = dist.get_world_size()
test_matmul_reducescatter(rank, world_size)
