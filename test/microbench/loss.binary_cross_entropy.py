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
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

shape_list = [(8733, 8733), (8733, 513), (513, 8733), (8192, 8192)]
backward = True


def _do_test(loss, input, target, dtype, device):
    output = loss(input, target)
    grad_output = torch.ones_like(output, dtype=dtype)
    grad_inputs = torch.autograd.grad(output, input, grad_output)

    return output, grad_inputs


def run_profile(
    loss, input, target, dtype, backward, cache_r, cache_w, device, num_iter
):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            cache_r = cache_w + 1
            _do_test(loss, input, target, dtype, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(loss, input, target, dtype, backward, cache_r, cache_w, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        cache_r = cache_w + 1
        _do_test(loss, input, target, dtype, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            M, N = shape[0], shape[1]
            input = torch.randn((M, N), requires_grad=True)
            target = torch.empty((M, N)).random_(2)
            cache_r = torch.randn(1024 * 1024 * 1024, device=args.device)
            cache_w = torch.randn(1024 * 1024 * 1024, device=args.device)
            for reduce in ["none", "mean", "sum"]:
                loss = nn.BCELoss(reduce=reduce)
                m = nn.Sigmoid()
                input = m(input).to(dtype=dtype, device=args.device)
                target = target.to(dtype=dtype, device=args.device)
                # warm up
                _do_test(loss, input, target, dtype, args.device)

                # go
                print(
                    "shape:",
                    (M, N),
                    "; datatype:",
                    dtype,
                    "; reduce:",
                    reduce,
                    "; backward:",
                    backward,
                )
                if not args.e2e_only:
                    run_profile(
                        loss,
                        input,
                        target,
                        dtype,
                        backward,
                        cache_r,
                        cache_w,
                        args.device,
                        args.num_iter,
                    )

                if not args.profile_only:
                    run_e2e(
                        loss,
                        input,
                        target,
                        dtype,
                        backward,
                        cache_r,
                        cache_w,
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
