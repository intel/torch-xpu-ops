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

# T,N,C,S
shape_list = [(32, 32, 32, 16), (128, 128, 128, 128), (8, 8, 4, 8)]
backward = True


def _test_loss_ctc(log_probs, targets, input_lengths, target_lengths, backward):
    loss_dpcpp = torch.nn.functional.ctc_loss(
        log_probs, targets, input_lengths, target_lengths
    )
    if backward:
        loss_dpcpp.backward()

def run_profile(log_probs, targets, input_lengths, target_lengths, backward, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU,
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            _test_loss_ctc(log_probs, targets, input_lengths, target_lengths, backward)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(log_probs, targets, input_lengths, target_lengths, backward, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        _test_loss_ctc(log_probs, targets, input_lengths, target_lengths, backward)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.float32]:
            T, N, C, S = shape[0], shape[1], shape[2], shape[3]
            g_cpu = torch.Generator()
            g_cpu.manual_seed(15)
            torch.manual_seed(15)
            log_probs = (
                torch.randn(T, N, C, dtype=dtype, device=args.device).log_softmax(2).detach().requires_grad_()
            )
            targets = torch.randint(1, N, (N, S), dtype=torch.long, device=args.device)
            input_lengths = torch.full((N,), T, dtype=torch.long, device=args.device)
            target_lengths = torch.randint(1, S, (N,), dtype=torch.long, device=args.device)

            if backward:
                log_probs.requires_grad_(True)

            # warm up
            _test_loss_ctc(log_probs, targets, input_lengths, target_lengths, backward)
            # go
            print(
                "shape:",
                (shape[0], shape[1], shape[2], shape[3]),
                "; datatype:",
                dtype,
                "; backward:",
                backward,
            )
            if not args.e2e_only:
                run_profile(log_probs, targets, input_lengths, target_lengths, backward, args.device, args.num_iter)

            if not args.profile_only:
                run_e2e(log_probs, targets, input_lengths, target_lengths, backward, args.device, args.num_iter)
            g_cpu = torch.Generator()
            g_cpu.manual_seed(15)
            torch.manual_seed(15)

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
