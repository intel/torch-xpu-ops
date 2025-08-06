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
    (1, 32, 128, 32, 32),  # all channel for 1 group
    (16, 1024, 128, 32, 32),  # normal shape, big memory
    (32, 32, 32, 64, 64),  # normal shape, small memory, 1 channel per group
    (32, 32, 512, 256, 256),  # group_num=32, channel for per group=16,big memory
    (8, 32, 32, 16, 64, 64),  # 3d
]
backward = True

def Group_norm(shape, dtype, channels_last, backward, affine, device):
    num_groups = shape[0]
    shape_input = (shape[1], shape[2], shape[3], shape[4])
    C = shape[2]
    memory_format = (
        torch.channels_last_3d
        if len(shape_input) == 5
        else torch.channels_last
    )

    if channels_last:
        input = (
            torch.randn(shape_input)
            .to(memory_format=memory_format)
            .to(device=device, dtype=dtype)
        )
    else:
        input = torch.randn(shape_input).to(device=device, dtype=dtype)

    if backward:
        input.requires_grad_(True)

    m = torch.nn.GroupNorm(num_groups, C, affine=affine, dtype=dtype).to(
        device
    )
    output = m(input)

    if backward:
        grad_out = torch.randn_like(output).to(device)
        (grad_dpcpp,) = torch.autograd.grad(output, input, grad_out)

def run_profile(shape, dtype, channels_last, backward, affine, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU, 
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(num_iter):
            Group_norm(shape, dtype, channels_last, backward, affine, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(shape, dtype, channels_last, backward, affine, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(num_iter):
        Group_norm(shape, dtype, channels_last, backward, affine, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                for affine in [False, True]:
                    # warm up
                    Group_norm(shape, dtype, channels_last, backward, affine, args.device)

                    # go
                    print(
                        "shape:",
                        (shape[1], shape[2], shape[3], shape[4]),
                        "; datatype:",
                        dtype,
                        "; channels_last:",
                        channels_last,
                        "; affine:",
                        affine,
                        "; backward:",
                        backward,
                    )
                    if not args.e2e_only:
                        run_profile(shape, dtype, channels_last, backward, affine, args.device, args.num_iter)

                    if not args.profile_only:
                        run_e2e(shape, dtype, channels_last, backward, affine, args.device, args.num_iter)

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
