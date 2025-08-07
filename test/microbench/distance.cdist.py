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
    ((8, 16), (2, 16)),
    ((10, 8192), (10, 8192)),
    ((10, 8192), (8192, 8192)),
    ((4, 512, 512), (4, 513, 512)),
    ((1, 512, 8192), (1, 1024, 8192)),
]
backward = True


def Cdist(input1, input2, backward, p, compute_mode, device):
    output = torch.cdist(input1, input2, p, compute_mode)
    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)


def run_profile(input1, input2, backward, p, compute_mode, device, num_iter):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Cdist(input1, input2, backward, p, compute_mode, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(input1, input2, backward, p, compute_mode, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Cdist(input1, input2, backward, p, compute_mode, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for shape in shape_list:
        for p in [0, 1, 2]:
            for compute_mode in [
                "use_mm_for_euclid_dist_if_necessary",
                "use_mm_for_euclid_dist",
                "donot_use_mm_for_euclid_dist",
            ]:
                for dtype in [torch.float32]:
                    input1 = torch.rand(shape[0], device=args.device, dtype=dtype)
                    input2 = torch.rand(shape[1], device=args.device, dtype=dtype)
                    if backward:
                        input1.requires_grad_(True)
                        input2.requires_grad_(True)
                    # warm up
                    Cdist(input1, input2, backward, p, compute_mode, args.device)

                    # go
                    print(
                        "shape:",
                        (shape),
                        "; datatype:",
                        dtype,
                        "; P:",
                        p,
                        "; mode:",
                        compute_mode,
                        "; backward:",
                        backward,
                    )

                    if not args.e2e_only:
                        run_profile(
                            input1,
                            input2,
                            backward,
                            p,
                            compute_mode,
                            args.device,
                            args.num_iter,
                        )

                    if not args.profile_only:
                        run_e2e(
                            input1,
                            input2,
                            backward,
                            p,
                            compute_mode,
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
