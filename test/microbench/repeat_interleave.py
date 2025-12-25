# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (16, 8, 23),
    (4, 2048, 2048),
]
backward = False


def Repeat_interleave(input, repeats, dim, backward, device):
    output = torch.repeat_interleave(input, repeats, dim)
    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)


def run_profile(input, repeats, dim, backward, device, num_iter):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Repeat_interleave(input, repeats, dim, backward, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(input, repeats, dim, backward, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Repeat_interleave(input, repeats, dim, backward, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for shape in shape_list:
        for repeats in [8]:
            for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                for dim in [0, 2]:
                    input = torch.randn(shape, device=args.device, dtype=dtype)

                    if backward:
                        input.requires_grad_(True)

                    # warm up
                    Repeat_interleave(input, repeats, dim, backward, args.device)

                    # go
                    print(
                        "shape:",
                        shape,
                        "; datatype:",
                        dtype,
                        "; dim:",
                        dim,
                        "; backward:",
                        backward,
                    )
                    if not args.e2e_only:
                        run_profile(
                            input, repeats, dim, backward, args.device, args.num_iter
                        )

                    if not args.profile_only:
                        run_e2e(
                            input, repeats, dim, backward, args.device, args.num_iter
                        )


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
