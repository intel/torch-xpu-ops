import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

forward_shape_list = [(2048, 256), (2048, 8192), (16, 8192 * 4)]
backward_shape_list = [(256, 256), (256, 8192), (16, 8192 * 4)]


def Pdist(input, backward, device):
    b = torch.nn.functional.pdist(input, 2)
    if backward:
        gy = torch.empty_like(b)
        b.backward(gy)


def run_profile(input, backward, device, num_iter):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Pdist(input, backward, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(input, backward, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Pdist(input, backward, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for backward in [False, True]:
        shape_list = backward_shape_list if backward else forward_shape_list
        for shape in shape_list:
            for dtype in [torch.float32]:
                input = torch.rand(shape, device=args.device, dtype=dtype)
                if backward:
                    input.requires_grad_(True)
                # warm up
                Pdist(input, backward, args.device)

                # go
                print("shape:", shape, "; datatype:", dtype, "; backward:", backward)
                if not args.e2e_only:
                    run_profile(input, backward, args.device, args.num_iter)

                if not args.profile_only:
                    run_e2e(input, backward, args.device, args.num_iter)


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
