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
    ((28, 4096, 9, 1), 2),  # LQCD shape
    ((512, 36, 4, 1), 2),
    ((4, 4096, 4096), 0),  # big shape
    ((2048, 4, 4096), 0),
    ((2048, 4096, 4), 0),
    ((2048, 4096, 4096), 0),
    ((4096, 8192, 8192), 0),
    ((4097, 8193, 8193), 0),
    ((4, 4096, 4096), 1),  # big shape
    ((2048, 4, 4096), 1),
    ((2048, 4096, 4), 1),
    ((2048, 4096, 4096), 1),
    ((4096, 8192, 8192), 1),
    ((4097, 8193, 8193), 1),
]
backward = False


def Scatter(shape, dtype, dim, g_xpu, device):
    if dim == 2:
        m, n, k1, k2 = shape[0][0], shape[0][1], shape[0][2], shape[0][3]
        src = torch.ones((m, n, k1), dtype=dtype, device=device)
        index = torch.randint(0, k2, (m, n, k1), generator=g_xpu, device=device)
        zeros = torch.zeros(m, n, k2, dtype=dtype, device=device)
    else:
        if dim == 0:
            m1, m2, n = shape[0][0], shape[0][1], shape[0][2]
            src = torch.ones((m1, n), dtype=dtype, device=device)
            index = torch.randint(0, m2, (m1, n), generator=g_xpu, device=device)
            zeros = torch.zeros(m2, n, dtype=src.dtype, device=device)
        else:
            m, n1, n2 = shape[0][0], shape[0][1], shape[0][2]
            src = torch.ones((m, n1), dtype=dtype, device=device)
            index = torch.randint(0, n2, (m, n1), generator=g_xpu, device=device)
            zeros = torch.zeros(m, n2, dtype=src.dtype, device=device)

    dst = zeros.scatter_(dim, index, src)


def run_profile(shape, dtype, dim, g_xpu, device, num_iter):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Scatter(shape, dtype, dim, g_xpu, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(shape, dtype, dim, g_xpu, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Scatter(shape, dtype, dim, g_xpu, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            dim = shape[1]
            g_xpu = torch.Generator(device=args.device)
            g_xpu.manual_seed(25)
            torch.manual_seed(25)
            # warm up
            Scatter(shape, dtype, dim, g_xpu, args.device)

            # go
            print(
                "shape:",
                shape[0],
                "; datatype:",
                dtype,
                "; dim:",
                dim,
                "; backward:",
                backward,
            )
            if not args.e2e_only:
                run_profile(shape, dtype, dim, g_xpu, args.device, args.num_iter)

            if not args.profile_only:
                run_e2e(shape, dtype, dim, g_xpu, args.device, args.num_iter)


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
