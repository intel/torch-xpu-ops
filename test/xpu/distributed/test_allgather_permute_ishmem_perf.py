"""
Performance benchmark for ISHMEM allgather+permute fusion.

Usage:
    mpirun -n 2 python test_allgather_permute_ishmem_perf.py

Optional environment variables:
    TOKENS_PER_RANK, HIDDEN_SIZE, TOPK, NUM_EXPERTS, LOOP, WARMUP, DTYPE
    RUN_REFERENCE=0 to skip the XCCL all_gather + local_permute_copy_ baseline.
"""

import os
import sys
import traceback

os.environ.setdefault("ISHMEM_SYMMETRIC_SIZE", str(512 * 1024 * 1024))

import torch
import torch.distributed as dist

from allgather_local_permute_fusion import (
    allgather_permute_ishmem,
    allgather_ishmem,
    compute_scatter_idx,
)

TOKENS_PER_RANK = int(os.environ.get("TOKENS_PER_RANK", 4096))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 5120))
TOPK = int(os.environ.get("TOPK", 8))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 128))
LOOP = int(os.environ.get("LOOP", 40))
WARMUP = int(os.environ.get("WARMUP", 20))
RUN_REFERENCE = os.environ.get("RUN_REFERENCE", "1") != "0"
RUN_NO_PERMUTE = os.environ.get("RUN_NO_PERMUTE", "1") != "0"
PCIE_DISCOUNT = 0.7
CROSS_GPU_BW_GBPS = 31.5 * PCIE_DISCOUNT
HBM_BW_GBPS = 437.0

if LOOP <= WARMUP:
    raise ValueError(f"LOOP ({LOOP}) must be greater than WARMUP ({WARMUP})")


def parse_dtype():
    dtype_name = os.environ.get("DTYPE", "bfloat16").lower()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_name in ("fp32", "float", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported DTYPE={dtype_name!r}; use bfloat16 or float32")


DTYPE = parse_dtype()


def bytes_to_mb(num_bytes):
    return num_bytes / (1024 * 1024)


def project_time_ms(bytes_count, bw_gbps):
    return bytes_count / (bw_gbps * 1e9) * 1e3


def init_distributed():
    os.environ["RANK"] = str(os.environ.get("PMI_RANK", 0))
    os.environ["WORLD_SIZE"] = str(os.environ.get("PMI_SIZE", 1))
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29526")
    if not dist.is_initialized():
        dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_id = rank % torch.xpu.device_count()
    torch.xpu.set_device(device_id)
    return rank, world_size, device_id


def make_inputs(rank, world_size, device):
    num_tokens = TOKENS_PER_RANK * world_size

    torch.manual_seed(1234 + rank)
    hidden_shard = torch.randn(
        TOKENS_PER_RANK,
        HIDDEN_SIZE,
        device=device,
        dtype=DTYPE,
    )

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(42)
    topk_idx_cpu = torch.randint(
        0,
        NUM_EXPERTS,
        (num_tokens, TOPK),
        generator=cpu_generator,
        dtype=torch.int32,
    )
    scatter_idx_cpu, _ = compute_scatter_idx(
        topk_idx_cpu,
        num_experts=NUM_EXPERTS,
    )
    scatter_idx = scatter_idx_cpu.to(device).contiguous()

    return hidden_shard, scatter_idx


def timed_loop(fn, loop, warmup):
    begin_events = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(loop)]

    for i in range(loop):
        if i >= warmup:
            begin_events[i].record()
        fn()
        if i >= warmup:
            end_events[i].record()

    torch.xpu.synchronize()
    dist.barrier()

    return [begin_events[i].elapsed_time(end_events[i]) for i in range(warmup, loop)]


def print_latency_summary(label, latencies, rank):
    avg = sum(latencies) / len(latencies)
    print(f"[{label} latency rank {rank}] {[f'{l:.3f}' for l in latencies]} ms")
    return avg


def run_reference_allgather_permute(
    hidden_shard,
    scatter_idx,
    output,
    gathered,
    group,
    world_size,
):
    if not hasattr(torch.ops.symm_mem, "local_permute_copy_"):
        raise RuntimeError(
            "local_permute_copy_ kernel is required for the reference path; "
            "build test/xpu/csrc/liblocal_permute_copy.so first or set RUN_REFERENCE=0"
        )

    dist.all_gather(gathered, hidden_shard, group=group)
    for src_rank in range(world_size):
        token_offset = src_rank * TOKENS_PER_RANK
        torch.ops.symm_mem.local_permute_copy_(
            gathered[src_rank],
            scatter_idx,
            token_offset,
            output,
        )


def benchmark():
    rank, world_size, device_id = init_distributed()
    print("start to init \n", flush=True)
    device = f"xpu:{device_id}"
    group = dist.group.WORLD
    num_tokens = TOKENS_PER_RANK * world_size

    hidden_shard, scatter_idx = make_inputs(rank, world_size, device)
    output_ishmem = torch.empty(
        (num_tokens * TOPK, HIDDEN_SIZE),
        device=device,
        dtype=hidden_shard.dtype,
    )

    def run_ishmem():
        allgather_permute_ishmem(
            hidden_shard,
            scatter_idx,
            output_ishmem,
            group=group,
        )
    print("!!!!!!!!!!!!!! start to run ishmem \n", flush=True)
    #for i in range(1):
    run_ishmem()
    run_ishmem()
    torch.xpu.synchronize()
    dist.barrier()
    print("!!!!!!!!!!!!!! finish to run ishmem \n", flush=True)

    ishmem_latencies = timed_loop(run_ishmem, LOOP, WARMUP)

    avg_ishmem = print_latency_summary(
        "ISHMEM allgather_permute",
        ishmem_latencies,
        rank,
    )

    output_ref = None
    ref_latencies = []
    if RUN_REFERENCE:
        output_ref = torch.empty_like(output_ishmem)
        gathered = [torch.empty_like(hidden_shard) for _ in range(world_size)]

        def run_reference():
            run_reference_allgather_permute(
                hidden_shard,
                scatter_idx,
                output_ref,
                gathered,
                group,
                world_size,
            )

        ref_latencies = timed_loop(run_reference, LOOP, WARMUP)
        avg_ref = print_latency_summary(
            "XCCL allgather + local_permute",
            ref_latencies,
            rank,
        )

        run_ishmem()
        run_reference()
        torch.xpu.synchronize()
        assert torch.equal(output_ishmem, output_ref), (
            f"ISHMEM allgather_permute mismatch in rank {rank}, "
            f"dtype={hidden_shard.dtype}, tokens_per_rank={TOKENS_PER_RANK}, "
            f"hidden={HIDDEN_SIZE}, topk={TOPK}"
        )
    else:
        avg_ref = None

    # ---- No-permute comparison: pure allgather (ISHMEM vs XCCL) ----
    avg_ishmem_ag = None
    avg_ref_ag = None
    ishmem_ag_latencies = []
    ref_ag_latencies = []
    if RUN_NO_PERMUTE:
        gathered_flat = torch.empty(
            (num_tokens, HIDDEN_SIZE),
            device=device,
            dtype=hidden_shard.dtype,
        )

        def run_ishmem_allgather():
            allgather_ishmem(hidden_shard, gathered_flat, group=group)

        # warmup + timed
        run_ishmem_allgather()
        run_ishmem_allgather()
        torch.xpu.synchronize()
        dist.barrier()
        ishmem_ag_latencies = timed_loop(run_ishmem_allgather, LOOP, WARMUP)
        avg_ishmem_ag = print_latency_summary(
            "ISHMEM allgather (no permute)", ishmem_ag_latencies, rank
        )

        gathered_list = [torch.empty_like(hidden_shard) for _ in range(world_size)]

        def run_xccl_allgather():
            dist.all_gather(gathered_list, hidden_shard, group=group)

        ref_ag_latencies = timed_loop(run_xccl_allgather, LOOP, WARMUP)
        avg_ref_ag = print_latency_summary(
            "XCCL all_gather (no permute)", ref_ag_latencies, rank
        )

    if rank == 0:
        elem_size = hidden_shard.element_size()
        allgather_bytes = (world_size - 1) * TOKENS_PER_RANK * HIDDEN_SIZE * elem_size
        permute_read_bytes = num_tokens * HIDDEN_SIZE * elem_size
        permute_write_bytes = num_tokens * TOPK * HIDDEN_SIZE * elem_size
        proj_ag_ms = project_time_ms(allgather_bytes, CROSS_GPU_BW_GBPS)
        proj_perm_ms = project_time_ms(
            permute_read_bytes + permute_write_bytes,
            HBM_BW_GBPS,
        )
        lower_bound = proj_ag_ms + proj_perm_ms

        print(f"\n{'=' * 72}")
        print("[ISHMEM Allgather + Permute]")
        print(
            f"  config: dtype={hidden_shard.dtype}, world_size={world_size}, "
            f"tokens_per_rank={TOKENS_PER_RANK}, hidden={HIDDEN_SIZE}, "
            f"topk={TOPK}, num_experts={NUM_EXPERTS}, loop={LOOP}, warmup={WARMUP}"
        )
        print(
            f"  ISHMEM: avg={avg_ishmem:.3f} ms  "
            f"min={min(ishmem_latencies):.3f} ms  max={max(ishmem_latencies):.3f} ms"
        )
        if avg_ref is not None:
            print(
                f"  XCCL reference: avg={avg_ref:.3f} ms  "
                f"min={min(ref_latencies):.3f} ms  max={max(ref_latencies):.3f} ms"
            )
            if avg_ref > 0:
                speedup = avg_ref / avg_ishmem
                print(f"  speedup_vs_reference={speedup:.2f}x")
        else:
            print("  XCCL reference: skipped (RUN_REFERENCE=0)")

        if avg_ishmem_ag is not None:
            print(f"\n  --- No-permute (pure allgather) ---")
            print(
                f"  ISHMEM allgather: avg={avg_ishmem_ag:.3f} ms  "
                f"min={min(ishmem_ag_latencies):.3f} ms  "
                f"max={max(ishmem_ag_latencies):.3f} ms"
            )
            print(
                f"  XCCL all_gather:  avg={avg_ref_ag:.3f} ms  "
                f"min={min(ref_ag_latencies):.3f} ms  "
                f"max={max(ref_ag_latencies):.3f} ms"
            )
            if avg_ref_ag > 0:
                print(
                    f"  speedup_vs_xccl (allgather only)="
                    f"{avg_ref_ag / avg_ishmem_ag:.2f}x"
                )

        print(
            f"\n  [Projection] allgather={bytes_to_mb(allgather_bytes):.1f} MB "
            f"@{CROSS_GPU_BW_GBPS:.1f} GB/s -> {proj_ag_ms:.3f} ms"
        )
        print(
            f"  [Projection] permute (R+W)="
            f"{bytes_to_mb(permute_read_bytes + permute_write_bytes):.1f} MB "
            f"@{HBM_BW_GBPS:.1f} GB/s -> {proj_perm_ms:.3f} ms"
        )
        print(f"  [Projection] lower_bound={lower_bound:.3f} ms")
        print(f"  [Efficiency] ishmem={lower_bound / avg_ishmem * 100:.1f}%")
        print(f"{'=' * 72}\n")


def main():
    exit_code = 0
    print("start to run", flush=True)
    try:
        benchmark()
        if dist.is_initialized():
            dist.barrier()
    except Exception:
        exit_code = 1
        traceback.print_exc()
    finally:
        # ishmem_finalize currently hangs in the mixed XCCL+ISHMEM test process.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
