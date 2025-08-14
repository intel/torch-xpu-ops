import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(1024, 1024)]
backward = False


def Index_fill(input, indices, dim, device):
    if dim == 0:
        output = input.index_fill(dim, indices, 1)
    else:
        output = input.index_fill(dim, indices, 2)

def run_profile(input, indices, dim, cache_r, cache_w, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU,
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            cache_r = cache_w * i
            Index_fill(input, indices, dim, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(input, indices, dim, cache_r, cache_w, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        cache_r = cache_w * i
        Index_fill(input, indices, dim, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for dim in [0, 1]:
                input = torch.zeros(shape, dtype=dtype, device=args.device)
                indices = torch.linspace(0, 1022, steps=512, device=args.device).to(torch.long)
                y_0 = torch.ones((512, 1024), dtype=dtype, device=args.device)
                y_1 = torch.randn((1024, 512), dtype=dtype, device=args.device)
                cache_r = torch.randn((1024 * 1024 * 1024), device=args.device)
                cache_w = torch.randn((1024 * 1024 * 1024), device=args.device)
                # warm up
                Index_fill(input, indices, dim, args.device)

                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; dim:",
                    dim,
                    "; backward:",
                    backward,
                )
                if not args.e2e_only:
                    run_profile(input, indices, dim, cache_r, cache_w, args.device, args.num_iter)

                if not args.profile_only:
                    run_e2e(input, indices, dim, cache_r, cache_w, args.device, args.num_iter)

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
