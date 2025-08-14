import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(1024, 1024, 1024), (6, 7, 3, 2), (8193, 8193, 4, 4)]
backward = True


def Remainder(input, divisor, backward, device):
    output = torch.remainder(input, divisor)
    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)

def run_profile(input, divisor, backward, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU,
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Remainder(input, divisor, backward, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(input, divisor, backward, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Remainder(input, divisor, backward, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for divisor in [2, -1.5, 3]:
                input = torch.randn(shape, device=args.device, dtype=dtype)
                if backward:
                    input.requires_grad_(True)

                # warm
                Remainder(input, divisor, backward, args.device)

                # go
                print(
                    "shape:",
                    shape[0],
                    "; datatype:",
                    dtype,
                    "; divisor:",
                    divisor,
                    "; backward:",
                    backward,
                )
                if not args.e2e_only:
                    run_profile(input, divisor, backward, args.device, args.num_iter)

                if not args.profile_only:
                    run_e2e(input, divisor, backward, args.device, args.num_iter)

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
