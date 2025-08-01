import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (8, 512, 7, 7, (1, 1)),
    (8, 512, 32, 32, (7, 7)),
    (8, 256, 56, 56, (14, 14)),
]


def adaptive_mp2d(shape, dtype, channels_last, backward):
    N, C, H, W, output_size = shape[0], shape[1], shape[2], shape[3], shape[4]

    if channels_last:
        input = (
            torch.randn(N, C, H, W)
            .to(memory_format=torch.channels_last)
            .to(device="xpu", dtype=dtype)
        )
    else:
        input = torch.randn(N, C, H, W).to(device="xpu", dtype=dtype)

    if backward:
        input.requires_grad_(True)
        Wout = output_size[0]
        Hout = output_size[1]
        grad = torch.randn([N, C, Hout, Wout]).to(device="xpu", dtype=dtype)

    adapt_mp2d = torch.nn.AdaptiveMaxPool2d(
        output_size=(Hout, Wout), return_indices=True
    )

    output = adapt_mp2d(input)

    if backward:
        output[0].backward(grad)


if __name__ == "__main__":
    backward = True
    num_iter = 20
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                # warm up
                adaptive_mp2d(shape, dtype, channels_last, backward)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3]),
                    "; datatype:",
                    dtype,
                    "; output_size:",
                    shape[4],
                    "; channels_last:",
                    channels_last,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(num_iter):
                        adaptive_mp2d(shape, dtype, channels_last, backward)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    adaptive_mp2d(shape, dtype, channels_last, backward)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
