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
    [1, 3, 1200, 1200],
    [1, 128, 1200, 1200],
    [1, 3, 1200, 1200],
    [128, 128, 5, 5],
    [8, 32, 256, 256],
]
scale_factor = [[3, 3], [3, 3], [7, 7], [7, 7], 3]
backward = True


def Bilinear2d(in_tensor, scale, backward, device):
    output = torch.nn.functional.interpolate(
        in_tensor,
        mode="bilinear",
        scale_factor=scale,
    )
    if backward:
        output = torch.autograd.grad(
            output, in_tensor, grad_outputs=torch.ones_like(output)
        )


def run_profile(in_tensor, scale, backward, cache_r, cache_w, device, num_iter):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(num_iter):
            cache_r = cache_w + 1
            Bilinear2d(in_tensor, scale, backward, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(in_tensor, scale, backward, cache_r, cache_w, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(num_iter):
        cache_r = cache_w + 1
        Bilinear2d(in_tensor, scale, backward, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for in_shape, scale in zip(shape_list, scale_factor):
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            in_tensor = torch.randn(
                in_shape, dtype=dtype, device=args.device, requires_grad=backward
            )
            cache_r = torch.randn((1024 * 1024 * 1024), device=args.device)
            cache_w = torch.randn((1024 * 1024 * 1024), device=args.device)
            # warm_up
            Bilinear2d(in_tensor, scale, backward, args.device)

            # go
            print(
                "shape:",
                (in_shape),
                "; datatype:",
                dtype,
                "; scale_factor:",
                scale,
                "; mode:",
                "bilinear",
                "; backward:",
                backward,
            )
            if not args.e2e_only:
                run_profile(
                    in_tensor,
                    scale,
                    backward,
                    cache_r,
                    cache_w,
                    args.device,
                    args.num_iter,
                )

            if not args.profile_only:
                run_e2e(
                    in_tensor,
                    scale,
                    backward,
                    cache_r,
                    cache_w,
                    args.device,
                    args.num_iter,
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
