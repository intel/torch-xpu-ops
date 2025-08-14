import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(1024, 8)]
backward = True
dict_len = 2500000
vect_len = 128
num_iter = 20


def Embedding(input, grad, emb, device):
    output = emb(input)
    output.backward(grad)

def run_profile(input, grad, emb, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU,
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Embedding(input, grad, emb, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(input, grad, emb, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Embedding(input, grad, emb, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            emb = torch.nn.Embedding(dict_len, vect_len, dtype=dtype, device=args.device)
            input = torch.randint(0, dict_len, (1024, 8), device=args.device)
            grad = torch.randn(1024, 8, vect_len, dtype=dtype, device=args.device)
            # warm up
            Embedding(input, grad, emb, args.device)

            # go
            print("shape:", (shape), "; datatype:", shape, dtype, "; backward:", backward)
            if not args.e2e_only:
                run_profile(input, grad, emb, args.device, args.num_iter)

            if not args.profile_only:
                run_e2e(input, grad, emb, args.device, args.num_iter)

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
