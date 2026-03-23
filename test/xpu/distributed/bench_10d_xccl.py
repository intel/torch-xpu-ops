#!/usr/bin/env python3
# Collective & P2P Operations Benchmark (device-agnostic: CUDA/NCCL, XPU/XCCL, etc.)
#
# Usage:
#   # Auto-detect accelerator and backend:
#   torchrun --standalone --nproc-per-node <N> bench_c10d_xccl.py
#
#   # Sweep world sizes (launches sub-processes internally):
#   python bench_c10d_xccl.py --sweep-world-sizes 2,4,8
#
#   # Select specific ops:
#   torchrun --standalone --nproc-per-node 2 bench_c10d_xccl.py --ops allreduce,allgather
#
#   # Export CSV:
#   torchrun --standalone --nproc-per-node 2 bench_c10d_xccl.py --export-csv results.csv

import argparse
import csv
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Device abstraction — works with any accelerator (CUDA, XPU, …)
# ---------------------------------------------------------------------------

def _get_device_type() -> str:
    """Return the current accelerator device type string, e.g. 'cuda' or 'xpu'."""
    acc = torch.accelerator.current_accelerator()
    if acc is not None:
        return acc.type
    # Fallback: try common accelerators
    for dt in ("cuda", "xpu"):
        if hasattr(torch, dt) and getattr(torch, dt).is_available():
            return dt
    raise RuntimeError("No accelerator available")


def _get_backend() -> str:
    """Return the default dist backend for the current accelerator."""
    return dist.get_default_backend_for_device(_get_device_type())


def _device_count() -> int:
    return torch.accelerator.device_count()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup() -> Tuple[int, int]:
    """Initialize process group and return (rank, world_size)."""
    if not dist.is_initialized():
        dist.init_process_group(backend=_get_backend())

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank % _device_count()
    torch.accelerator.set_device_index(local_rank)
    return rank, world_size


def _device(rank: int) -> torch.device:
    device_type = _get_device_type()
    return torch.device(f"{device_type}:{rank % _device_count()}")


def _sync_device(device: torch.device):
    torch.accelerator.synchronize(device)


def _measure_time(
    fn: Callable,
    num_iters: int,
    num_warmup: int,
    device: torch.device,
) -> float:
    """Measure average execution time in milliseconds using accelerator events."""
    # Warmup
    for _ in range(num_warmup):
        fn()
    _sync_device(device)
    dist.barrier()

    device_module = getattr(torch, device.type)
    start_event = device_module.Event(enable_timing=True)
    end_event = device_module.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iters):
        fn()
    end_event.record()
    _sync_device(device)

    elapsed_ms = start_event.elapsed_time(end_event) / num_iters
    return elapsed_ms


def _algbw(nbytes: int, time_ms: float) -> float:
    """Algorithm bandwidth in GB/s."""
    if time_ms <= 0:
        return 0.0
    return nbytes / (time_ms * 1e-3) / 1e9


def _busbw(nbytes: int, time_ms: float, world_size: int, op: str) -> float:
    """Bus bandwidth in GB/s (accounts for ring/tree factors)."""
    algbw = _algbw(nbytes, time_ms)
    n = world_size
    # Bus bandwidth correction factors (same as nccl-tests)
    factors = {
        "broadcast": 1.0,
        "reduce": 1.0,
        "allreduce": 2.0 * (n - 1) / n,
        "allgather": (n - 1) / n,
        "reduce_scatter": (n - 1) / n,
        "alltoall": (n - 1) / n,
        "alltoall_single": (n - 1) / n,
        "gather": 1.0,
        "scatter": 1.0,
        "send_recv": 1.0,
        "batch_isend_irecv": 1.0,
        "barrier": 0.0,
    }
    return algbw * factors.get(op, 1.0)


# ---------------------------------------------------------------------------
# Benchmark functions for each operation
# ---------------------------------------------------------------------------

def bench_broadcast(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel = nbytes // dtype_size(dtype)
    tensor = torch.randn(numel, dtype=dtype, device=device)

    for root in range(min(world_size, 2)):  # bench root=0 and root=1 if available
        def fn():
            dist.broadcast(tensor, src=root)

        time_ms = _measure_time(fn, num_iters, num_warmup, device)
        yield {
            "op": "broadcast",
            "root": root,
            "time_ms": time_ms,
            "nbytes": nbytes,
            "algbw_GBs": _algbw(nbytes, time_ms),
            "busbw_GBs": _busbw(nbytes, time_ms, world_size, "broadcast"),
        }


def bench_allreduce(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel = nbytes // dtype_size(dtype)
    tensor = torch.randn(numel, dtype=dtype, device=device)

    def fn():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    yield {
        "op": "allreduce",
        "time_ms": time_ms,
        "nbytes": nbytes,
        "algbw_GBs": _algbw(nbytes, time_ms),
        "busbw_GBs": _busbw(nbytes, time_ms, world_size, "allreduce"),
    }


def bench_reduce(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel = nbytes // dtype_size(dtype)
    tensor = torch.randn(numel, dtype=dtype, device=device)

    for root in range(min(world_size, 2)):
        def fn(root=root):
            dist.reduce(tensor, dst=root, op=dist.ReduceOp.SUM)

        time_ms = _measure_time(fn, num_iters, num_warmup, device)
        yield {
            "op": "reduce",
            "root": root,
            "time_ms": time_ms,
            "nbytes": nbytes,
            "algbw_GBs": _algbw(nbytes, time_ms),
            "busbw_GBs": _busbw(nbytes, time_ms, world_size, "reduce"),
        }


def bench_allgather(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    # Each rank has nbytes/world_size data, output is nbytes total
    numel_per_rank = max(1, nbytes // (dtype_size(dtype) * world_size))
    input_tensor = torch.randn(numel_per_rank, dtype=dtype, device=device)
    output_tensors = [
        torch.empty(numel_per_rank, dtype=dtype, device=device)
        for _ in range(world_size)
    ]

    def fn():
        dist.all_gather(output_tensors, input_tensor)

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    total_bytes = numel_per_rank * world_size * dtype_size(dtype)
    yield {
        "op": "allgather",
        "time_ms": time_ms,
        "nbytes": total_bytes,
        "algbw_GBs": _algbw(total_bytes, time_ms),
        "busbw_GBs": _busbw(total_bytes, time_ms, world_size, "allgather"),
    }


def bench_reduce_scatter(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel_per_rank = max(1, nbytes // (dtype_size(dtype) * world_size))
    input_tensor = torch.randn(numel_per_rank * world_size, dtype=dtype, device=device)
    output_tensor = torch.empty(numel_per_rank, dtype=dtype, device=device)

    def fn():
        dist.reduce_scatter_tensor(output_tensor, input_tensor, op=dist.ReduceOp.SUM)

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    total_bytes = numel_per_rank * world_size * dtype_size(dtype)
    yield {
        "op": "reduce_scatter",
        "time_ms": time_ms,
        "nbytes": total_bytes,
        "algbw_GBs": _algbw(total_bytes, time_ms),
        "busbw_GBs": _busbw(total_bytes, time_ms, world_size, "reduce_scatter"),
    }


def bench_alltoall_single(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel = max(world_size, nbytes // dtype_size(dtype))
    # Make divisible by world_size
    numel = (numel // world_size) * world_size
    input_tensor = torch.randn(numel, dtype=dtype, device=device)
    output_tensor = torch.empty(numel, dtype=dtype, device=device)

    def fn():
        dist.all_to_all_single(output_tensor, input_tensor)

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    total_bytes = numel * dtype_size(dtype)
    yield {
        "op": "alltoall_single",
        "time_ms": time_ms,
        "nbytes": total_bytes,
        "algbw_GBs": _algbw(total_bytes, time_ms),
        "busbw_GBs": _busbw(total_bytes, time_ms, world_size, "alltoall_single"),
    }


def bench_alltoall(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel_per_rank = max(1, nbytes // (dtype_size(dtype) * world_size))
    input_tensors = [
        torch.randn(numel_per_rank, dtype=dtype, device=device)
        for _ in range(world_size)
    ]
    output_tensors = [
        torch.empty(numel_per_rank, dtype=dtype, device=device)
        for _ in range(world_size)
    ]

    def fn():
        dist.all_to_all(output_tensors, input_tensors)

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    total_bytes = numel_per_rank * world_size * dtype_size(dtype)
    yield {
        "op": "alltoall",
        "time_ms": time_ms,
        "nbytes": total_bytes,
        "algbw_GBs": _algbw(total_bytes, time_ms),
        "busbw_GBs": _busbw(total_bytes, time_ms, world_size, "alltoall"),
    }


def bench_gather(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel = nbytes // dtype_size(dtype)
    input_tensor = torch.randn(numel, dtype=dtype, device=device)
    root = 0

    if rank == root:
        gather_list = [
            torch.empty(numel, dtype=dtype, device=device) for _ in range(world_size)
        ]
    else:
        gather_list = None

    def fn():
        dist.gather(input_tensor, gather_list, dst=root)

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    total_bytes = numel * world_size * dtype_size(dtype)
    yield {
        "op": "gather",
        "root": root,
        "time_ms": time_ms,
        "nbytes": total_bytes,
        "algbw_GBs": _algbw(total_bytes, time_ms),
        "busbw_GBs": _busbw(total_bytes, time_ms, world_size, "gather"),
    }


def bench_scatter(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel = nbytes // dtype_size(dtype)
    output_tensor = torch.empty(numel, dtype=dtype, device=device)
    root = 0

    if rank == root:
        scatter_list = [
            torch.randn(numel, dtype=dtype, device=device) for _ in range(world_size)
        ]
    else:
        scatter_list = None

    def fn():
        dist.scatter(output_tensor, scatter_list, src=root)

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    total_bytes = numel * world_size * dtype_size(dtype)
    yield {
        "op": "scatter",
        "root": root,
        "time_ms": time_ms,
        "nbytes": total_bytes,
        "algbw_GBs": _algbw(total_bytes, time_ms),
        "busbw_GBs": _busbw(total_bytes, time_ms, world_size, "scatter"),
    }


def bench_send_recv(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel = nbytes // dtype_size(dtype)
    tensor = torch.randn(numel, dtype=dtype, device=device)

    if world_size < 2:
        return

    # Pair ranks: 0<->1, 2<->3, ...
    if rank % 2 == 0 and rank + 1 < world_size:
        peer = rank + 1
    elif rank % 2 == 1:
        peer = rank - 1
    else:
        # Odd rank out, skip
        dist.barrier()
        return

    def fn():
        if rank < peer:
            dist.send(tensor, dst=peer)
            dist.recv(tensor, src=peer)
        else:
            dist.recv(tensor, src=peer)
            dist.send(tensor, dst=peer)

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    yield {
        "op": "send_recv",
        "time_ms": time_ms,
        "nbytes": nbytes,
        "algbw_GBs": _algbw(nbytes, time_ms),
        "busbw_GBs": _busbw(nbytes, time_ms, world_size, "send_recv"),
    }


def bench_batch_isend_irecv(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    numel = nbytes // dtype_size(dtype)
    send_tensor = torch.randn(numel, dtype=dtype, device=device)
    recv_tensor = torch.empty(numel, dtype=dtype, device=device)

    if world_size < 2:
        return

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    def fn():
        send_op = dist.P2POp(dist.isend, send_tensor, next_rank)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, prev_rank)
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    yield {
        "op": "batch_isend_irecv",
        "time_ms": time_ms,
        "nbytes": nbytes,
        "algbw_GBs": _algbw(nbytes, time_ms),
        "busbw_GBs": _busbw(nbytes, time_ms, world_size, "batch_isend_irecv"),
    }


def bench_barrier(rank, world_size, device, nbytes, dtype, num_iters, num_warmup):
    def fn():
        dist.barrier()

    time_ms = _measure_time(fn, num_iters, num_warmup, device)
    yield {
        "op": "barrier",
        "time_ms": time_ms,
        "nbytes": 0,
        "algbw_GBs": 0.0,
        "busbw_GBs": 0.0,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OP_REGISTRY = {
    "broadcast": bench_broadcast,
    "allreduce": bench_allreduce,
    "reduce": bench_reduce,
    "allgather": bench_allgather,
    "reduce_scatter": bench_reduce_scatter,
    "alltoall_single": bench_alltoall_single,
    "alltoall": bench_alltoall,
    "gather": bench_gather,
    "scatter": bench_scatter,
    "send_recv": bench_send_recv,
    "batch_isend_irecv": bench_batch_isend_irecv,
    "barrier": bench_barrier,
}

ALL_OPS = list(OP_REGISTRY.keys())

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
}


def dtype_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


# ---------------------------------------------------------------------------
# Main benchmark driver
# ---------------------------------------------------------------------------

def _generate_sizes(min_exp: int, max_exp: int, steps: int) -> List[int]:
    """Generate message sizes in bytes (power-of-2 spaced)."""
    if steps <= 1:
        return [2 ** max_exp]
    sizes = []
    for i in range(steps):
        exp = min_exp + (max_exp - min_exp) * i / (steps - 1)
        size = int(2 ** exp)
        # Round to 256-byte boundary for alignment
        size = max(256, (size + 255) // 256 * 256)
        if size not in sizes:
            sizes.append(size)
    return sizes


def run_benchmark(args):
    rank, world_size = _setup()
    device = _device(rank)

    ops = args.ops.split(",") if args.ops else ALL_OPS
    dtype = DTYPE_MAP[args.dtype]
    sizes = _generate_sizes(args.min_size, args.max_size, args.size_steps)

    all_results = []

    device_type = _get_device_type()
    backend = _get_backend()

    if rank == 0:
        print(f"\n{'='*90}")
        print(f"  Collective Benchmark — {device_type.upper()}/{backend.upper()}")
        print(f"  world_size={world_size}, dtype={args.dtype}")
        print(f"  iters={args.num_iters}, warmup={args.num_warmup}")
        print(f"  message sizes: {len(sizes)} points from 2^{args.min_size} to 2^{args.max_size} bytes")
        print(f"  ops: {', '.join(ops)}")
        print(f"{'='*90}\n")

    for op_name in ops:
        if op_name not in OP_REGISTRY:
            if rank == 0:
                print(f"  [WARN] Unknown op '{op_name}', skipping.")
            continue

        bench_fn = OP_REGISTRY[op_name]

        if rank == 0:
            print(f"  {'Op':<22} {'Size':>12} {'Time(ms)':>12} {'AlgBW(GB/s)':>14} {'BusBW(GB/s)':>14}")
            print(f"  {'-'*22} {'-'*12} {'-'*12} {'-'*14} {'-'*14}")

        # barrier op doesn't need size sweep
        if op_name == "barrier":
            for result in bench_fn(rank, world_size, device, 0, dtype, args.num_iters, args.num_warmup):
                result["world_size"] = world_size
                result["dtype"] = args.dtype
                result["backend"] = backend
                result["device_type"] = device_type
                all_results.append(result)
                if rank == 0:
                    _print_row(result)
        else:
            for nbytes in sizes:
                for result in bench_fn(rank, world_size, device, nbytes, dtype, args.num_iters, args.num_warmup):
                    result["world_size"] = world_size
                    result["dtype"] = args.dtype
                    result["backend"] = backend
                    result["device_type"] = device_type
                    all_results.append(result)
                    if rank == 0:
                        _print_row(result)

        if rank == 0:
            print()

    # Export CSV
    if rank == 0 and args.export_csv:
        _export_csv(all_results, args.export_csv)
        print(f"  Results exported to {args.export_csv}")

    dist.barrier()
    dist.destroy_process_group()


def _print_row(result: Dict[str, Any]):
    op = result["op"]
    extra = ""
    if "root" in result:
        extra = f"(root={result['root']})"

    nbytes = result.get("nbytes", 0)
    size_str = _format_size(nbytes) if nbytes > 0 else "N/A"

    print(
        f"  {op + extra:<22} {size_str:>12} "
        f"{result['time_ms']:>12.4f} "
        f"{result['algbw_GBs']:>14.3f} "
        f"{result['busbw_GBs']:>14.3f}"
    )


def _format_size(nbytes: int) -> str:
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.2f} GB"
    elif nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.2f} MB"
    elif nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.2f} KB"
    else:
        return f"{nbytes} B"


def _export_csv(results: List[Dict[str, Any]], filepath: str):
    if not results:
        return
    keys = list(results[0].keys())
    # Ensure consistent columns across all rows
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    keys = sorted(all_keys)

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# World-size sweep launcher
# ---------------------------------------------------------------------------

def sweep_world_sizes(args):
    """Launch the benchmark for each requested world size using torchrun."""
    world_sizes = [int(x) for x in args.sweep_world_sizes.split(",")]
    script = os.path.abspath(__file__)

    for ws in world_sizes:
        ndevices = _device_count()
        if ws > ndevices:
            print(f"[SKIP] world_size={ws} > available devices ({ndevices})")
            continue

        csv_arg = []
        if args.export_csv:
            base, ext = os.path.splitext(args.export_csv)
            csv_arg = ["--export-csv", f"{base}_ws{ws}{ext}"]

        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={ws}",
            script,
            "--ops", args.ops or ",".join(ALL_OPS),
            "--dtype", args.dtype,
            "--num-iters", str(args.num_iters),
            "--num-warmup", str(args.num_warmup),
            "--min-size", str(args.min_size),
            "--max-size", str(args.max_size),
            "--size-steps", str(args.size_steps),
        ] + csv_arg

        print(f"\n{'#'*90}")
        print(f"# Launching world_size={ws}")
        print(f"# Command: {' '.join(cmd)}")
        print(f"{'#'*90}\n")

        subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collective & P2P Operations Benchmark (NCCL / XCCL / …)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available ops: {', '.join(ALL_OPS)}

Examples:
  # Run all ops with 2 GPUs (auto-detect CUDA/XPU):
  torchrun --standalone --nproc-per-node 2 bench_c10d_xccl.py

  # Only allreduce and allgather, export CSV:
  torchrun --standalone --nproc-per-node 2 bench_c10d_xccl.py \\
      --ops allreduce,allgather --export-csv results.csv

  # Sweep world sizes 2,4,8:
  python bench_c10d_xccl.py --sweep-world-sizes 2,4,8

  # Large message sizes with bfloat16:
  torchrun --standalone --nproc-per-node 4 bench_c10d_xccl.py \\
      --dtype bfloat16 --min-size 20 --max-size 30
""",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help=f"Comma-separated ops to benchmark (default: all). Choices: {','.join(ALL_OPS)}",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=list(DTYPE_MAP.keys()),
        help="Data type for tensors (default: float32)",
    )
    parser.add_argument("--num-iters", type=int, default=50, help="Number of measured iterations (default: 50)")
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument(
        "--min-size",
        type=int,
        default=10,
        help="Min message size as 2^N bytes (default: 10 → 1KB)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=28,
        help="Max message size as 2^N bytes (default: 28 → 256MB)",
    )
    parser.add_argument("--size-steps", type=int, default=10, help="Number of size steps (default: 10)")
    parser.add_argument("--export-csv", type=str, default=None, help="Export results to CSV file")
    parser.add_argument(
        "--sweep-world-sizes",
        type=str,
        default=None,
        help="Comma-separated world sizes to sweep, e.g. '2,4,8'. "
             "Launches separate torchrun processes for each.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.sweep_world_sizes:
        sweep_world_sizes(args)
    else:
        run_benchmark(args)


if __name__ == "__main__":
    main()
