import time
import argparse
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

shape_list = [(8192, 8192)]
backward = True


def Margin_loss(loss, input, target, reduce, dtype, device):
    output = loss(input, target)
    if reduce == "none":
        output.backward(torch.ones_like(output, dtype=dtype).to(device))
    else:
        output.backward(torch.tensor((1.0), dtype=dtype).to(device))

def run_profile(loss, input, target, reduce, dtype, cache_r, cache_w, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU,
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            cache_r = cache_w
            Margin_loss(loss, input, target, reduce, dtype, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(loss, input, target, reduce, dtype, cache_r, cache_w, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        cache_r = cache_w
        Margin_loss(loss, input, target, reduce, dtype, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for reduce in ["none", "mean"]:
                input = torch.randn(shape, dtype=dtype, device=args.device)
                target = torch.randn(shape, dtype=dtype, device=args.device).long()
                input.requires_grad = True
                loss = nn.MultiLabelMarginLoss(reduction=reduce)

                cache_r = torch.randn((1024 * 1024 * 1024), device=args.device)
                cache_w = torch.randn((1024 * 1024 * 1024), device=args.device)

                # warm up
                Margin_loss(loss, input, target, reduce, dtype, args.device)

                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; backward:",
                    backward,
                    "; reduce: 0" if (reduce == "none") else "; reduce: 1",
                )
                if not args.e2e_only:
                    run_profile(loss, input, target, reduce, dtype, cache_r, cache_w, args.device, args.num_iter)

                if not args.profile_only:
                    run_e2e(loss, input, target, reduce, dtype, cache_r, cache_w, args.device, args.num_iter)

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
