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
    (16, 24, 28, 19, 19, (3), (2)),
    (16, 1984, 7, 7, 7, (3, 2, 2), (2, 1, 2)),
    (64, 1024, 14, 14, 14, (6), (4)),
]
backward = True


def AVGPool3d(shape, dtype, channels_last, backward, device):
    N, C, D, H, W, kernel_size, stride = (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
        shape[6],
    )

    if channels_last:
        input = (
            torch.randn(N, C, D, H, W)
            .to(memory_format=torch.channels_last_3d)
            .to(device=device, dtype=dtype)
        )
    else:
        input = torch.randn(N, C, D, H, W).to(device=device, dtype=dtype)

    if backward:
        input.requires_grad_(True)
        if isinstance(kernel_size, int):
            Dout = (D - kernel_size) / stride + 1
            Hout = (H - kernel_size) / stride + 1
            Wout = (W - kernel_size) / stride + 1
        else:
            Dout = (D - kernel_size[0]) / stride[0] + 1
            Hout = (H - kernel_size[1]) / stride[1] + 1
            Wout = (W - kernel_size[2]) / stride[2] + 1
        grad = torch.randn([C, int(Dout), int(Hout), int(Wout)]).to(
            device=device, dtype=dtype
        )

    AVG3d = torch.nn.AvgPool3d(shape[5], stride=shape[6])

    output = AVG3d(input)

    if backward:
        output[0].backward(grad)

def run_profile(shape, dtype, channels_last, backward, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU, 
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(num_iter):
            AVGPool3d(shape, dtype, channels_last, backward, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(shape, dtype, channels_last, backward, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(num_iter):
        AVGPool3d(shape, dtype, channels_last, backward, device)
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
                AVGPool3d(shape, dtype, channels_last, backward, args.device)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3], shape[4]),
                    "; datatype:",
                    dtype,
                    "; kernel_size:",
                    shape[5],
                    "; stride:",
                    shape[6],
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
