import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [((25, 300), (22, 300), (15, 300)), ((2, 1000), (100, 1000), (8192, 1000))]
backward = False


def Pad_sequence(a, b, c, batch_first, padding_value, dtype, backward, device):
    output = torch.nn.utils.rnn.pad_sequence(
        ([a, b, c]), batch_first, padding_value
    )
    if backward:
        gy = torch.empty_like(output)
        output.backward(gy)

def run_profile(a, b, c, batch_first, padding_value, dtype, backward, device, num_iter):
    with profile(
        activities=[ProfilerActivity.CPU, 
                  ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for i in range(num_iter):
            Pad_sequence(a, b, c, batch_first, padding_value, dtype, backward, device)
    print(prof.key_averages().table(sort_by="{}_time_total".format(device)))

def run_e2e(a, b, c, batch_first, padding_value, dtype, backward, device, num_iter):
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        Pad_sequence(a, b, c, batch_first, padding_value, dtype, backward, device)
    if device in ['xpu', 'cuda']:
        torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")

def benchmark(args):
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for batch_first in [False, True]:
                for padding_value in [0.0, 1.0, 2.0]:
                    a = torch.randn(shape[0], device=args.device, dtype=dtype)
                    b = torch.randn(shape[1], device=args.device, dtype=dtype)
                    c = torch.randn(shape[2], device=args.device, dtype=dtype)

                    if backward:
                        a.requires_grad_(True)
                        b.requires_grad_(True)
                        c.requires_grad_(True)

                    # warm up
                    Pad_sequence(a, b, c, batch_first, padding_value, dtype, backward, args.device)

                    # go
                    print(
                        "shape:",
                        (shape),
                        "; datatype:",
                        dtype,
                        "; batch_first:",
                        batch_first,
                        "; padding_value:",
                        padding_value,
                        "; backward:",
                        backward,
                    )
                    if not args.e2e_only:
                        run_profile(a, b, c, batch_first, padding_value, dtype, backward, args.device, args.num_iter)

                    if not args.profile_only:
                        run_e2e(a, b, c, batch_first, padding_value, dtype, backward, args.device, args.num_iter)

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
