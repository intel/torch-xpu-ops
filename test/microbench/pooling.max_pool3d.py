import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (16, 32, 64, 64, 64, 32, 32, 32),
    (1, 4, 144, 144, 144, 72, 72, 72),
    (512, 512, 12, 12, 12, 6, 6, 6),
]
backward = True


def mp3d(shape, dtype, channels_last, backward, device):
    torch.manual_seed(20)
    N, C, H, W, D, oH, oW, oD = (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
        shape[6],
        shape[7],
    )

    if channels_last:
        input = (
            torch.randn(N, C, H, W, D)
            .to(memory_format=torch.channels_last_3d)
            .to(device=device, dtype=dtype)
        )
    else:
        input = torch.randn(N, C, H, W, D).to(device=device, dtype=dtype)

    if backward:
        input.requires_grad_(True)
        grad = torch.randn([N, C, oH, oW, oD]).to(device=device, dtype=dtype)

    fmp = torch.nn.MaxPool3d(2, return_indices=True)
    output = fmp(input)


def run_profile(shape, dtype, channels_last, backward, device, num_iter):
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.XPU if device == "xpu" else ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            mp3d(shape, dtype, channels_last, backward, device)
    print(prof.key_averages().table(sort_by=f"{device}_time_total"))


def run_e2e(shape, dtype, channels_last, backward, device, num_iter):
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        mp3d(shape, dtype, channels_last, backward, device)
    if device in ["xpu", "cuda"]:
        torch.xpu.synchronize() if device == "xpu" else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                # warm up
                mp3d(shape, dtype, channels_last, backward, args.device)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3], shape[4]),
                    "; datatype:",
                    dtype,
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
                        args.device,
                        args.num_iter,
                    )

                if not args.profile_only:
                    run_e2e(
                        shape,
                        dtype,
                        channels_last,
                        backward,
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
