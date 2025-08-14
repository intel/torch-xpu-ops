import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (4, 64, 128, 128),
    (4, 65, 128, 128),
    (8, 128, 128, 128),
]
backward = True


def maxUnpool2d(shape, dtype, channels_last, backward, device):
    N, C, H, W = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
    kernel_size = 2

    pool = torch.nn.MaxPool2d(kernel_size, return_indices=True).to(
        device=device, dtype=dtype
    )
    unpool = torch.nn.MaxUnpool2d(kernel_size).to(device=device, dtype=dtype)
    torch.manual_seed(20)

    if channels_last:
        input = (
            torch.randn([N, C, H, W])
            .to(memory_format=torch.channels_last)
            .to(device=device, dtype=dtype)
        )
    else:
        input = torch.randn([N, C, H, W]).to(device=device, dtype=dtype)
    output, indices = pool(input)

    if channels_last:
        x_dpcpp = output.to(memory_format=torch.channels_last).to(
            device=device, dtype=dtype
        )
        indices_dpcpp = indices.to(memory_format=torch.channels_last).to(
            device=device, dtype=torch.int64
        )
    else:
        x_dpcpp = output.to(device=device, dtype=dtype)
        indices_dpcpp = indices.to(device=device, dtype=torch.int64)

    if backward:
        x_dpcpp.requires_grad_(True)
        if channels_last:
            grad_dpcpp = (
                torch.randn([N, C, H, W])
                .to(memory_format=torch.channels_last)
                .to(device=device, dtype=dtype)
            )
        else:
            grad_dpcpp = torch.randn([N, C, H, W]).to(device=device, dtype=dtype)

    y_dpcpp = unpool(x_dpcpp, indices_dpcpp, output_size=torch.Size([N, C, H, W]))

    if backward:
        y_dpcpp.backward(grad_dpcpp)

def run_profile(shape, dtype, channels_last, backward, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU,
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            maxUnpool2d(shape, dtype, channels_last, backward, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(shape, dtype, channels_last, backward, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        maxUnpool2d(shape, dtype, channels_last, backward, device)
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
                maxUnpool2d(shape, dtype, channels_last, backward, args.device)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3]),
                    "; datatype:",
                    dtype,
                    "; kernel_size:",
                    str(2),
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
