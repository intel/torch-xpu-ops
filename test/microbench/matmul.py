import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (4, 4096, 50400),
    (4, 2048, 32000),
    (4, 4096, 128256),
    (4, 5120, 32000),
    (4, 3072, 32064),
    (4, 4096, 50272),
    (4, 4096, 250880),
    (4, 2560, 32000),
    (4, 2048, 50272),
    (4, 1792, 250880),
]
backward = True


def matmul(m, n, k, dtype, backward, device):
    m1 = torch.rand(2, m, k).type(dtype).to(device)
    m2 = torch.rand(k, n).type(dtype).to(device)
    if backward:
        m1.requires_grad_(True)
        m2.requires_grad_(True)
    output = torch.matmul(m1, m2)

    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)

def run_profile(shape, dtype, backward, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU, 
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            matmul(shape[0], shape[2], shape[1], dtype, backward, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(shape, dtype, backward, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        matmul(shape[0], shape[2], shape[1], dtype, backward, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            # warm up
            matmul(shape[0], shape[2], shape[1], dtype, backward, args.device)

            # go
            print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
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
