import time
import argparse
import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(4, 15000)]
backward = False


def parse_args():
    parser = argparse.ArgumentParser(description="OP Benchmark")
    parser.add_argument("--device", type=str, default="xpu", help='Device to run on (e.g., "cpu", "cuda", "xpu")')
    parser.add_argument("--num_iter", type=int, default=20, help='Number of iterations')
    parser.add_argument("--profile-only", action="store_true", help='Only Run profile timing')
    parser.add_argument("--e2e-only", action="store_true", help='Only Run E2E timing')

    args = parser.parse_args()
    return args

def benchmark(shape, dtype, mode, device, num_iter, do_profile, do_e2e):
    d = torch.rand(shape, dtype=dtype, device=device)
    e = torch.rand(shape, dtype=dtype, device=device)
    f = d < e
    g = e[f]

    # warm up
    if mode == "with_nonzero":
        for i in range(100):
            f = d < e
            g = e[f]
    else:
        f = torch.linspace(0, 4 - 2, steps=int(4 / 2), device=device).to(
            torch.long
        )
        for i in range(100):
            g = e[f]

    # go
    print(
        "shape:",
        (shape),
        "; datatype:",
        dtype,
        "; mode:",
        mode,
        "; backward:",
        backward,
    )
    if not do_e2e:
        with profile(
            activities=[ProfilerActivity.CPU, 
                ProfilerActivity.XPU if device == 'xpu' else ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for i in range(num_iter):
                if mode == "with_nonzero":
                    f = d < e
                    g = e[f]
                else:
                    g = e[f]
        print(prof.key_averages().table(sort_by="xpu_time_total"))

    # E2E time
    if not do_profile:
        if device in ['xpu', 'cuda']:
            torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            if mode == "with_nonzero":
                df = d < e
                g = e[f]
            else:
                g = e[f]
        if device in ['xpu', 'cuda']:
            torch.xpu.synchronize() if device == 'xpu' else torch.cuda.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")

def main():
    args = parse_args()
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for mode in ["with_nonzero", "without_nonzero"]:
                benchmark(
                    shape=shape,
                    dtype=dtype,
                    mode=mode,
                    device=args.device,
                    num_iter=args.num_iter,
                    do_profile=args.profile_only,
                    do_e2e=args.e2e_only,
                )

if __name__ == "__main__":
    main()
