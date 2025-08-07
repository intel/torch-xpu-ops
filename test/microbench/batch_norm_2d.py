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
    (256, 256, 56, 56, 256),
    (256, 2048, 7, 7, 2048),
    (24, 512, 28, 28, 512),
    (24, 1024, 14, 14, 1024),
    (4, 8, 640, 1024, 8),
    (4, 48, 20, 32, 48),
]
backward = True


def BTN2d(shape, dtype, channels_last, backward, device):
    N, C, H, W, num_features = shape[0], shape[1], shape[2], shape[3], shape[4]

    if channels_last:
        input = (
            torch.randn(N, C, H, W)
            .to(memory_format=torch.channels_last)
            .to(device="xpu", dtype=dtype)
        )
    else:
        input = torch.randn(N, C, H, W).to(device="xpu", dtype=dtype)

    if backward:
        input.requires_grad_(True)
        grad = torch.randn([C, H, W]).to(device="xpu", dtype=dtype)

    BTN = torch.nn.BatchNorm2d(shape[4], device=device)

    output = BTN(input)

    if backward:
        output[0].backward(grad)

def run_profile(shape, dtype, channels_last, backward, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU,
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            BTN2d(shape, dtype, channels_last, backward, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(shape, dtype, channels_last, backward, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        BTN2d(shape, dtype, channels_last, backward, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                # warm up
                BTN2d(shape, dtype, channels_last, backward, args.device)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3]),
                    "; datatype:",
                    dtype,
                    "; num_features:",
                    shape[4],
                    "; channels_last:",
                    channels_last,
                    "; backward:",
                    backward,
                )

                if not args.e2e_only:
                    run_profile(shape, dtype, channels_last, backward, args.device, args.num_iter)

                if not args.profile_only:
                    run_e2e(shape, dtype, channels_last, backward, args.device, args.num_iter)

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
