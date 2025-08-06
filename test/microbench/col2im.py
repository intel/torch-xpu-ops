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
    ((1, 147, 1359556), (1200, 1200)),
    ((1, 147, 36100), (224, 224)),
    ((1, 147, 33814), (63, 1200)),
    ((1, 147, 33814), (1200, 63)),
]
kernel_size = (7, 7)
dilation = (6, 6)
backward = True

def Col2im(shape, dtype, backward, device):
    input = torch.randn(shape[0], dtype=dtype, device=device, requires_grad=True)
    if backward:
        input.requires_grad_(True)
    output_size = shape[1]

    output = torch.nn.functional.fold(
        input, output_size, kernel_size, dilation, 1, 1
    )
    if backward:
        torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output))

def run_profile(shape, dtype, backward, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU, 
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(num_iter):
            Col2im(shape, dtype, backward, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(shape, dtype, backward, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(num_iter):
        Col2im(shape, dtype, backward, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            # warm up
            Col2im(shape, dtype, backward, args.device)

            # go
            print(
                "shape:",
                shape[0],
                "; datatype:",
                dtype,
                "; output_size:",
                shape[1],
                "; backward:",
                backward,
            )
            if not args.e2e_only:
                run_profile(shape, dtype, backward, args.device, args.num_iter)

            if not args.profile_only:
                run_e2e(shape, dtype, backward, args.device, args.num_iter)

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
