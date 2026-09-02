"""Correctness and performance test for the RDMA + NVL dispatcher.

The lower and upper rank halves are two IPC-capable switch domains. A token
sent inside its source domain uses a direct ISHMEM IPC pointer; a token sent
to the other domain is packed and transferred by RDMA.
"""

import json
import os
import sys
import time
from contextlib import nullcontext

os.environ.setdefault("ISHMEM_IB_ENABLE_IBGDA", "1")
os.environ.setdefault("ISHMEM_IBGDA_DIRECT_DOORBELL", "1")
os.environ.setdefault("ISHMEM_ENABLE_GPU_IPC", "1")
os.environ.setdefault("ISHMEM_ENABLE_ACCESSIBLE_HOST_HEAP", "0")
os.environ.setdefault("ISHMEM_SYMMETRIC_SIZE", str(512 * 1024 * 1024))

import torch
import torch.distributed as dist


WORLD_SIZE = int(os.environ.get("PMI_SIZE", os.environ.get("WORLD_SIZE", 4)))
assert WORLD_SIZE in {4, 8}, "This test requires 4 or 8 ranks"
RANKS_PER_SWITCH = WORLD_SIZE // 2
NUM_TOKENS = int(os.environ.get("NUM_TOKENS", 4096))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 4096))
TOPK = int(os.environ.get("TOPK", 8))
EXPERTS_PER_RANK = int(os.environ.get("EXPERTS_PER_RANK", 64))
CAPACITY = int(os.environ.get("NUM_MAX_TOKENS_PER_RANK", NUM_TOKENS))
CHANNELS = int(os.environ.get("DISPATCH_RDMA_NVL_CHANNELS", 16))
WARMUP = int(os.environ.get("WARMUP", 10))
LOOP = int(os.environ.get("LOOP", 30))
SEED = int(os.environ.get("SEED", 1234))
ENABLE_PROFILE = os.environ.get("ENABLE_PROFILE", "1") != "0"
KERNEL_NAME = "DispatchRdmaNvlKernel"

_HERE = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_HERE, "..", "csrc")


def load_extension():
    path = os.path.join(_CSRC, "libdispatch_rdma_nvl.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run bash 4rank-rdma.sh")
    torch.ops.load_library(path)


def parse_dtype():
    name = os.environ.get("DTYPE", "bfloat16").lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "half", "float16"}:
        return torch.float16
    if name in {"fp32", "float", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported DTYPE={name}")


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29547")
    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == WORLD_SIZE
    device_index = rank % torch.xpu.device_count()
    torch.xpu.set_device(device_index)
    return rank, f"xpu:{device_index}"


def profiled_latencies(trace_path, expected):
    with open(trace_path, "r", encoding="utf-8") as trace_file:
        events = json.load(trace_file).get("traceEvents", [])
    kernel_events = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("cat") == "kernel"
    ]
    unexpected = [
        event.get("name", "")
        for event in kernel_events
        if KERNEL_NAME not in event.get("name", "")
    ]
    if unexpected:
        raise RuntimeError(
            "dispatch_rdma_nvl must launch only DispatchRdmaNvlKernel; "
            f"found additional kernels: {sorted(set(unexpected))}"
        )
    if len(kernel_events) != expected:
        raise RuntimeError(
            f"Expected exactly {expected} XPU kernel events, "
            f"found {len(kernel_events)}"
        )
    values = [
        float(event["dur"]) / 1000.0
        for event in kernel_events
        if KERNEL_NAME in event.get("name", "")
        and "dur" in event
    ]
    if len(values) != expected:
        raise RuntimeError(
            f"Expected {expected} {KERNEL_NAME} events, found {len(values)}"
        )
    return values


def payload_layout(hidden_size, element_size, topk):
    hidden_bytes = hidden_size * element_size
    src_rank_offset = (hidden_bytes + 3) // 4 * 4
    topk_idx_offset = (src_rank_offset + 4 + 15) // 16 * 16
    topk_weights_offset = topk_idx_offset + topk * 8
    bytes_per_token = (topk_weights_offset + topk * 4 + 15) // 16 * 16
    return (
        hidden_bytes,
        src_rank_offset,
        topk_idx_offset,
        topk_weights_offset,
        bytes_per_token,
    )


def main():
    rank, device = init_distributed()
    load_extension()
    dtype = parse_dtype()
    assert CAPACITY >= NUM_TOKENS, (
        "The correctness test requires NUM_MAX_TOKENS_PER_RANK >= NUM_TOKENS"
    )

    torch.manual_seed(SEED + rank)
    x = torch.randn(NUM_TOKENS, HIDDEN_SIZE, dtype=dtype, device=device)
    num_experts = WORLD_SIZE * EXPERTS_PER_RANK
    assert TOPK <= num_experts
    topk_idx = (
        torch.rand(NUM_TOKENS, num_experts, device=device)
        .argsort(dim=1)[:, :TOPK]
        .to(torch.int64)
        .contiguous()
    )
    topk_weights = torch.rand(
        NUM_TOKENS, TOPK, dtype=torch.float32, device=device
    )
    is_token_in_rank = torch.zeros(
        NUM_TOKENS, WORLD_SIZE, dtype=torch.bool, device=device
    )
    destination_ranks = topk_idx // EXPERTS_PER_RANK
    is_token_in_rank.scatter_(1, destination_ranks, True)

    (
        hidden_bytes,
        src_rank_offset,
        topk_idx_offset,
        topk_weights_offset,
        bytes_per_token,
    ) = payload_layout(HIDDEN_SIZE, x.element_size(), TOPK)
    recv_payload = torch.zeros(
        WORLD_SIZE,
        CAPACITY,
        bytes_per_token,
        dtype=torch.uint8,
        device=device,
    )
    recv_channel_counts = torch.zeros(
        WORLD_SIZE, CHANNELS, dtype=torch.int64, device=device
    )

    def run():
        torch.ops.symm_mem.dispatch_rdma_nvl(
            x,
            topk_idx,
            topk_weights,
            is_token_in_rank,
            recv_payload,
            recv_channel_counts,
            rank,
            WORLD_SIZE,
            CAPACITY,
            CHANNELS,
        )

    run()
    torch.xpu.synchronize()

    all_x = [torch.empty_like(x) for _ in range(WORLD_SIZE)]
    dist.all_gather(all_x, x)
    all_topk_idx_i32 = [
        torch.empty_like(topk_idx, dtype=torch.int32) for _ in range(WORLD_SIZE)
    ]
    dist.all_gather(all_topk_idx_i32, topk_idx.to(torch.int32))
    all_topk_idx = [value.to(torch.int64) for value in all_topk_idx_i32]
    all_topk_weights = [
        torch.empty_like(topk_weights) for _ in range(WORLD_SIZE)
    ]
    dist.all_gather(all_topk_weights, topk_weights)
    all_routes = [
        torch.empty_like(is_token_in_rank) for _ in range(WORLD_SIZE)
    ]
    dist.all_gather(all_routes, is_token_in_rank)

    for src in range(WORLD_SIZE):
        for channel in range(CHANNELS):
            token_begin = NUM_TOKENS * channel // CHANNELS
            token_end = NUM_TOKENS * (channel + 1) // CHANNELS
            slot_begin = token_begin
            route_slice = all_routes[src][token_begin:token_end, rank]
            selected = (
                route_slice.nonzero(as_tuple=True)[0] + token_begin
            )
            count = selected.numel()
            got_count = int(recv_channel_counts[src, channel].item())
            assert got_count == count, (
                f"[rank {rank}] source {src} channel {channel} count "
                f"mismatch: got {got_count}, expected {count}"
            )
            if count == 0:
                continue
            output_slice = slice(slot_begin, slot_begin + count)
            payload = recv_payload[src, output_slice].cpu()
            got_x = (
                payload[:, :hidden_bytes]
                .contiguous()
                .view(dtype)
                .view(count, HIDDEN_SIZE)
            )
            got_src_rank = (
                payload[
                    :,
                    src_rank_offset : src_rank_offset + 4,
                ]
                .contiguous()
                .view(torch.int32)
                .view(count)
            )
            got_topk_idx = (
                payload[
                    :,
                    topk_idx_offset : topk_idx_offset + TOPK * 8,
                ]
                .contiguous()
                .view(torch.int64)
                .view(count, TOPK)
            )
            got_topk_weights = (
                payload[
                    :,
                    topk_weights_offset : topk_weights_offset + TOPK * 4,
                ]
                .contiguous()
                .view(torch.float32)
                .view(count, TOPK)
            )
            assert torch.equal(
                got_src_rank,
                torch.full((count,), src, dtype=torch.int32),
            )
            assert torch.equal(
                got_topk_idx,
                all_topk_idx[src][selected].cpu(),
            )
            assert torch.allclose(
                got_topk_weights,
                all_topk_weights[src][selected].cpu(),
                atol=1e-6,
                rtol=0,
            )
            assert torch.allclose(
                got_x.float(),
                all_x[src][selected].float().cpu(),
                atol=1e-3,
                rtol=0,
            )

    print(
        f"[rank {rank}] dispatch_rdma_nvl correctness OK "
        f"counts={recv_channel_counts.sum(dim=1).tolist()}",
        flush=True,
    )

    for _ in range(WARMUP):
        run()
    torch.xpu.synchronize()
    timed_iters = max(LOOP - WARMUP, 1)
    start = time.time()
    for _ in range(timed_iters):
        run()
    torch.xpu.synchronize()
    wall_ms = (time.time() - start) * 1000.0 / timed_iters

    profiler = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        )
        if ENABLE_PROFILE
        else nullcontext()
    )
    if ENABLE_PROFILE:
        with profiler:
            for _ in range(timed_iters):
                run()
            torch.xpu.synchronize()

    logical_bytes_per_token = (
        hidden_bytes + 4 + TOPK * 8 + TOPK * 4
    )
    local_begin = (rank // RANKS_PER_SWITCH) * RANKS_PER_SWITCH
    ipc_mask = torch.zeros(WORLD_SIZE, dtype=torch.bool, device=device)
    ipc_mask[local_begin : local_begin + RANKS_PER_SWITCH] = True
    ipc_mask[rank] = False
    ipc_tokens = int(is_token_in_rank[:, ipc_mask].sum().item())
    remote_mask = torch.ones(WORLD_SIZE, dtype=torch.bool, device=device)
    remote_mask[local_begin : local_begin + RANKS_PER_SWITCH] = False
    rdma_tokens = int(is_token_in_rank[:, remote_mask].sum().item())
    dispatched_copies = int(is_token_in_rank.sum().item())
    algorithm_bytes = dispatched_copies * logical_bytes_per_token
    ipc_bytes = ipc_tokens * bytes_per_token
    rdma_bytes = rdma_tokens * bytes_per_token
    transport_bytes = ipc_bytes + rdma_bytes

    if ENABLE_PROFILE:
        trace_path = f"profile_dispatch_rdma_nvl_rank{rank}.json"
        profiler.export_chrome_trace(trace_path)
        latencies = profiled_latencies(trace_path, timed_iters)
        kernel_ms = sum(latencies) / len(latencies)
        print(
            f"[rank {rank}] [{KERNEL_NAME}] avg={kernel_ms:.3f} ms "
            f"min={min(latencies):.3f} ms max={max(latencies):.3f} ms "
            f"algorithm BW~={algorithm_bytes / 1e6 / kernel_ms:.2f} GB/s "
            f"IPC bytes/kernel time~={ipc_bytes / 1e6 / kernel_ms:.2f} GB/s "
            f"RDMA bytes/kernel time~={rdma_bytes / 1e6 / kernel_ms:.2f} GB/s "
            f"combined payload/kernel time~="
            f"{transport_bytes / 1e6 / kernel_ms:.2f} GB/s",
            flush=True,
        )

    print(
        f"[rank {rank}] end2end avg={wall_ms:.3f} ms "
        f"algorithm BW~={algorithm_bytes / 1e6 / wall_ms:.2f} GB/s",
        flush=True,
    )

    torch.ops.symm_mem.dispatch_rdma_nvl_finalize(
        torch.empty(0, device=device)
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
