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
    (8, 32, 256, 256, (3)),
    (8, 512, 16, 16, (1.5)),
    (16, 1024, 23, 23, (2.3)),
    (4, 32, 80, 128, (2)),
]
backward = True


def Interpolate2d(shape, dtype, channels_last, backward, mode, device):
    N, C, H, W, scale_factor = shape[0], shape[1], shape[2], shape[3], shape[4]

    if channels_last:
        input = (
            torch.randn(N, C, H, W, requires_grad=True)
            .to(memory_format=torch.channels_last)
            .to(device=device, dtype=dtype)
        )
    else:
        input = torch.randn(N, C, H, W, requires_grad=True).to(
            device=device, dtype=dtype
        )

    output = torch.nn.functional.interpolate(input, scale_factor=shape[4], mode=mode)

    if backward:
        output.backward(torch.ones_like(output))


def run_profile(shape, dtype, channels_last, backward, mode, device, num_iter):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Interpolate2d(shape, dtype, channels_last, backward, mode, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(shape, dtype, channels_last, backward, mode, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Interpolate2d(shape, dtype, channels_last, backward, mode, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                for mode in ["nearest-exact"]:
                    # warm up
                    Interpolate2d(
                        shape, dtype, channels_last, backward, mode, args.device
                    )

                    # go
                    print(
                        "shape:",
                        (shape[0], shape[1], shape[2], shape[3]),
                        "; datatype:",
                        dtype,
                        "; scale_factor:",
                        shape[4],
                        "; mode:",
                        mode,
                        "; channels_last:",
                        channels_last,
                        "; backward:",
                        backward,
                    )
                    if not args.e2e_only:
                        run_profile(
                            shape,
                            dtype,
                            channels_last,
                            backward,
                            mode,
                            args.device,
                            args.num_iter,
                        )

                    if not args.profile_only:
                        run_e2e(
                            shape,
                            dtype,
                            channels_last,
                            backward,
                            mode,
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
