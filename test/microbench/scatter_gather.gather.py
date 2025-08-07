# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    ((2048, 64, 4), (2048, 64, 1), 2),  # LQCD shape
    ((28, 4096, 9), (28, 4096, 1), 2),
    ((512, 36, 4), (512, 36, 1), 2),
    ((102400 * 6400, 4), (102400 * 6400, 1), 1),  # big shape thin
    ((102400, 4 * 6400), (25600, 4 * 6400), 0),  # big shape fat
    ((4 * 6400, 102400), (1 * 6400, 102400), 0),
    ((10240, 8192), (10240, 2048), 1),  # medium shape
    ((8192, 10240), (2048, 2560), 1),
    ((10240, 8192), (2560, 8192), 0),
    ((8192, 10240), (2048, 10240), 0),
]
backward = False


def Gather(a, dim, index, device):
    torch.gather(a, dim, index)

def run_profile(a, dim, index, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU,
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Gather(a, dim, index, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(a, dim, index, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Gather(a, dim, index, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            shapes = shape[0]
            ishapes = shape[1]
            dim = shape[2]
            g_xpu = torch.Generator(device=args.device)
            g_xpu.manual_seed(25)
            torch.manual_seed(25)
            a = torch.randn(shapes, dtype=dtype, device=args.device)
            index = torch.randint(1, shapes[dim], ishapes, device=args.device, generator=g_xpu)

            # warm up
            Gather(a, dim, index, args.device)

            # go
            print(
                "shape:",
                shapes,
                "; kernel_size:",
                ishapes,
                "; datatype:",
                dtype,
                "; dim:",
                dim,
                "; backward:",
                backward,
            )
            if not args.e2e_only:
                run_profile(a, dim, index, args.device, args.num_iter)

            if not args.profile_only:
                run_e2e(a, dim, index, args.device, args.num_iter)

def parse_args():
    parser = argparse.ArgumentParser(description='OP Benchmark')
    parser.add_argument('--device', type=str, default='xpu',
                        help='Device to run on (e.g., "cpu", "cuda", "xpu")')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--profile-only', action='store_true',
                       help='Only Run profile timing')
    group.add_argument('--e2e-only', action='store_true',
                       help='Only Run E2E timing')
    parser.add_argument('--num-iter', type=int, default=20,
                        help='Number of iterations')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    benchmark(args)
