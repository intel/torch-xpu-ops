import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"

shape_list = [
    (8, 512, 32, 32, (7, 7)),
    (8, 256, 56, 56, (14, 14)),
]
num_iter = 20


def Adaptive_AVGPool2d(shape, dtype, channels_last, backward):
    N, C, H, W, output_size = (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
    )

    if channels_last:
        input = (
            torch.randn(N, C, H, W, requires_grad=True)
            .to(memory_format=torch.channels_last)
            .to(device=device, dtype=dtype)
        )
    else:
        input = torch.randn(N, C, H, W, requires_grad=True).to(
            device=device, dtype=dtype
        )

    if backward:
        input.requires_grad_(True)
        Wout = output_size[0]
        Hout = output_size[1]
        grad = torch.rand([C, Hout, Wout], requires_grad=True).to(
            device=device, dtype=dtype
        )

    AdaptAVG2d = torch.nn.AdaptiveAvgPool2d(shape[4])

    output = AdaptAVG2d(input)

    if backward:
        output[0].backward(grad)


if __name__ == "__main__":
    backward = True
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                # warm up
                Adaptive_AVGPool2d(shape, dtype, channels_last, backward)

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
                        Adaptive_AVGPool2d(shape, dtype, channels_last, backward)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    Adaptive_AVGPool2d(shape, dtype, channels_last, backward)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
