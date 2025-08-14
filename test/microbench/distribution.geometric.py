import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(8192, 8192)]
backward = False


def Geometric(input, device):
    input.geometric_(0.5)


def run_profile(input, device, num_iter):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Geometric(input, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(input, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Geometric(input, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            input = torch.randn(shape, dtype=torch.bfloat16, device=args.device)
            # warm up
            Geometric(input, args.device)

            # go
            print(
                "shape:",
                (shape),
                "; datatype:",
                dtype,
                "; P:",
                0.5,
                "; backward:",
                backward,
            )
            if not args.e2e_only:
                run_profile(input, args.device, args.num_iter)

            if not args.profile_only:
                run_e2e(input, args.device, args.num_iter)


def parse_args():
    parser = argparse.ArgumentParser(description="OP Benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default="xpu",
        help='Device to run on (e.g., "cpu", "cuda", "xpu")',
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--profile-only", action="store_true", help="Only Run profile timing"
    )
    group.add_argument("--e2e-only", action="store_true", help="Only Run E2E timing")
    parser.add_argument("--num-iter", type=int, default=20, help="Number of iterations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    benchmark(args)
