import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(8192, 8192), (16, 1024)]
backward = True


def Dropout(shape, dtype, backward, device):
    H, W = (shape[0], shape[1])
    input = torch.randn((H, W)).to(dtype=dtype, device=device)

    dropout = torch.nn.Dropout(p=0.5)
    dropout.to(device=device, dtype=dtype)
    if backward:
        grad_dpcpp = torch.randn((H, W)).to(device=device, dtype=dtype)
        input.requires_grad_(True)

    # warm up
    output = dropout(input)
    if backward:
        output.backward(grad_dpcpp)

def run_profile(shape, dtype, backward, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU, 
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(num_iter):
            Dropout(shape, dtype, backward, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(shape, dtype, backward, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(num_iter):
        Dropout(shape, dtype, backward, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            # warm up
            Dropout(shape, dtype, backward, args.device)

            # go
            print(
                "shape:",
                (shape),
                "; datatype:",
                dtype,
                "; P:",
                0.5,
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
