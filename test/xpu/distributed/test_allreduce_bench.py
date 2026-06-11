import os
import json
import torch
import torch.distributed as dist
from torch.profiler import profile, ProfilerActivity



os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29800'


def log_rank0(msg):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(msg, flush=True)


def fmt_size(num_bytes):
    if num_bytes >= (1 << 30):
        return f"{num_bytes / (1 << 30):.2f} GB"
    if num_bytes >= (1 << 20):
        return f"{num_bytes / (1 << 20):.2f} MB"
    if num_bytes >= (1 << 10):
        return f"{num_bytes / (1 << 10):.2f} KB"
    return f"{num_bytes:.0f} B"


def busbw_factor(world_size):
    return 2.0 * (world_size - 1.0) / world_size


def algbw_gbs(num_bytes, latency_us):
    if latency_us <= 0:
        return 0.0
    return num_bytes / (latency_us * 1e-6) / 1e9


def print_summary_table(rows):
    if not rows:
        return

    print("\n[summary-table] allreduce trace metrics", flush=True)
    print(
        "| Size | avg_us | min_us | max_us | var_us | busBW(GB/s) | minAvg(last50)_us | busBW_minAvg(GB/s) |",
        flush=True,
    )
    print(
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        flush=True,
    )
    for r in rows:
        print(
            f"| {r['size']} | {r['avg_us']:.2f} | {r['min_us']:.2f} | {r['max_us']:.2f} | {r['var_us']:.2f} | "
            f"{r['bus_bw']:.3f} | {r['min_avg_us']:.2f} | {r['bus_bw_min']:.3f} |",
            flush=True,
        )


def print_iteration_table(last_matrix, measure):
    if not last_matrix:
        return

    print("\n[last-50-table] rows=rank, cols=iteration-id, value=kernel_us", flush=True)
    header = ["rank/iter"] + [str(i) for i in range(measure)]
    print("| " + " | ".join(header) + " |", flush=True)
    print("|" + "---|" * len(header), flush=True)

    for rank_id, vals in enumerate(last_matrix):
        row_vals = [f"{v:.2f}" for v in vals]
        print("| " + " | ".join([str(rank_id)] + row_vals) + " |", flush=True)


def parse_kernel_durations_from_trace(trace_path, keyword):
    with open(trace_path, "r", encoding="utf-8") as f:
        trace = json.load(f)

    events = trace.get("traceEvents", [])
    durations_us = []
    keyword_lower = keyword.lower()
    for ev in events:
        name = str(ev.get("name", ""))
        if keyword_lower in name.lower() and "dur" in ev:
            try:
                durations_us.append(float(ev["dur"]))
            except (TypeError, ValueError):
                continue
    return durations_us


def init_dist():
    dist.init_process_group(backend="xccl")
    local_rank = dist.get_rank()
    torch.xpu.set_device(local_rank)
    log_rank0(
        f"[init] world_size={dist.get_world_size()}, rank={dist.get_rank()}, xpu={local_rank}"
    )
    return local_rank


def benchmark_allreduce(
    size_kb=128,
    warmup=20,
    loop=100,
    measure_last=50,
    kernel_keyword="oneccl_allreduce_pcie",
    trace_prefix="allreduce_bench_trace",
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    size_bytes = size_kb * 1024
    measure = min(loop, measure_last)

    if rank == 0:
        print(
            f"[benchmark] start size={size_kb} KB, warmup={warmup}, loop={loop}, measure(last)={measure}",
            flush=True,
        )

    numel = size_kb * 1024 // 2  # fp16

    tensor_vec = [
        torch.rand(
            numel,
            device="xpu",
            dtype=torch.float16,
        )
        for _ in range(loop)
    ]

    # warmup: fully synchronous path
    for _ in range(warmup):
        dist.all_reduce(tensor_vec[0])
    torch.xpu.synchronize()

    dist.barrier()

    trace_path = f"{trace_prefix}_size{size_kb}_rank{rank}.json"

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        for i in range(loop):
            dist.all_reduce(tensor_vec[i])
        torch.xpu.synchronize()

    dist.barrier()
    prof.export_chrome_trace(trace_path)
    dist.barrier()

    all_kernel_us = parse_kernel_durations_from_trace(trace_path, kernel_keyword)
    if len(all_kernel_us) < measure:
        raise RuntimeError(
            f"rank {rank}: keyword '{kernel_keyword}' matched {len(all_kernel_us)} events, "
            f"expected at least {measure}. trace={trace_path}"
        )
    per_us = all_kernel_us[-measure:]

    per_us_tensor = torch.tensor(per_us, dtype=torch.float64, device="xpu")
    gathered = [torch.empty_like(per_us_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, per_us_tensor)

    if rank == 0:
        gathered_cpu = [x.cpu() for x in gathered]
        stacked = torch.stack(gathered_cpu, dim=0)

        iter_min = torch.min(stacked, dim=0).values
        iter_max = torch.max(stacked, dim=0).values
        iter_avg = torch.mean(stacked, dim=0)

        avg_us = float(torch.mean(iter_avg).item())
        min_us = float(torch.mean(iter_min).item())
        max_us = float(torch.mean(iter_max).item())
        var_us = float(torch.mean(iter_max - iter_min).item())
        min_avg_us = float(torch.mean(iter_min).item())

        bus_bw = algbw_gbs(size_bytes, avg_us) * busbw_factor(world_size)
        bus_bw_min = algbw_gbs(size_bytes, min_avg_us) * busbw_factor(world_size)

        print(f"[benchmark] done size={size_kb} KB", flush=True)

        return {
            "size": fmt_size(size_bytes),
            "avg_us": avg_us,
            "min_us": min_us,
            "max_us": max_us,
            "var_us": var_us,
            "bus_bw": bus_bw,
            "min_avg_us": min_avg_us,
            "bus_bw_min": bus_bw_min,
            "matrix": [vals.tolist() for vals in gathered_cpu],
            "measure": measure,
        }

    return None


def main():
    init_dist()

    summary_rows = []
    last_matrix = None
    last_measure = 0

    # Test sizes from 8KB to 8MB
    sizes_kb = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    for size_kb in sizes_kb:
        result = benchmark_allreduce(
            size_kb=size_kb,
            warmup=20,
            loop=100,
            measure_last=50,
        )

        if dist.get_rank() == 0 and result is not None:
            summary_rows.append({
                "size": result["size"],
                "avg_us": result["avg_us"],
                "min_us": result["min_us"],
                "max_us": result["max_us"],
                "var_us": result["var_us"],
                "bus_bw": result["bus_bw"],
                "min_avg_us": result["min_avg_us"],
                "bus_bw_min": result["bus_bw_min"],
            })
            last_matrix = result["matrix"]
            last_measure = result["measure"]

    if dist.get_rank() == 0:
        print_summary_table(summary_rows)
        print_iteration_table(last_matrix, last_measure)

    log_rank0("[done] destroy process group")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
