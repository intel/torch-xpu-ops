# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

import random
import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

dict_len = 2500000
vect_len = 128
batch = 1024
backward = True


def Embedding_bag(dtype, backward, reduce, device):
    input = torch.empty([batch], dtype=torch.long, device=device)
    emb = torch.nn.EmbeddingBag(
        dict_len, vect_len, mode=reduce, dtype=dtype, device=device
    )
    for i in range(batch):
        input[i] = random.randint(0, dict_len - 1)

    bag = torch.empty([batch], dtype=torch.long, device=device)
    for i in range(batch):
        bag[i] = i

    if backward:
        grad = torch.randn(batch, vect_len, dtype=dtype, device=device)

    output = emb(input, bag)
    if backward:
        output.backward(grad)

def run_profile(dtype, backward, reduce, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU, 
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(num_iter):
            Embedding_bag(dtype, backward, reduce, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(dtype, backward, reduce, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(num_iter):
        Embedding_bag(dtype, backward, reduce, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for reduce in ["max", "mean", "sum"]:
            # warm up
            Embedding_bag(dtype, backward, reduce, args.device)

            # go
            print(
                "shape:",
                (batch),
                "; datatype:",
                dtype,
                "; reduce:",
                reduce,
                "; backward:",
                backward,
            )
            if not args.e2e_only:
                run_profile(dtype, backward, reduce, args.device, args.num_iter)

            if not args.profile_only:
                run_e2e(dtype, backward, reduce, args.device, args.num_iter)

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
