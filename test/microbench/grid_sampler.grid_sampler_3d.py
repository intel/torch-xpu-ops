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
    (2, 5, 6, 3, 5),
    (8, 16, 64, 64, 64),
]
backward = True


def Grad_sample3d(input, grid, backward, mode, padding_mode, align_corners, device):
    output = torch.nn.functional.grid_sample(
        input,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    if backward:
        output.sum().backward()


def run_profile(
    input, grid, backward, mode, padding_mode, align_corners, device, num_iter
):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Grad_sample3d(
                input, grid, backward, mode, padding_mode, align_corners, device
            )
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(input, grid, backward, mode, padding_mode, align_corners, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Grad_sample3d(input, grid, backward, mode, padding_mode, align_corners, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for mode in ["bilinear", "nearest"]:
                for padding_mode in ["zeros", "border", "reflection"]:
                    for align_corners in [True, False]:
                        N, C, D, H, W = shape
                        input = torch.randn(
                            N, C, D, H, W, dtype=dtype, device=args.device
                        )
                        grid = torch.randn(
                            N, D, H, W, 3, dtype=dtype, device=args.device
                        )

                        if backward:
                            input.requires_grad_(True)
                            grid.requires_grad_(True)
                        # warm up
                        Grad_sample3d(
                            input,
                            grid,
                            backward,
                            mode,
                            padding_mode,
                            align_corners,
                            args.device,
                        )

                        # go
                        print(
                            "shape:",
                            (shape),
                            "; datatype:",
                            dtype,
                            "; mode:",
                            mode,
                            "; padding_mode:",
                            padding_mode,
                            "; align_corners:",
                            align_corners,
                            "; backward:",
                            backward,
                        )
                        if not args.e2e_only:
                            run_profile(
                                input,
                                grid,
                                backward,
                                mode,
                                padding_mode,
                                align_corners,
                                args.device,
                                args.num_iter,
                            )

                        if not args.profile_only:
                            run_e2e(
                                input,
                                grid,
                                backward,
                                mode,
                                padding_mode,
                                align_corners,
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
