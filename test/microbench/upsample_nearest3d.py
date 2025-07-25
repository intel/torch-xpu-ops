import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
num_iter = 20

shape_list = [
    (8, 32, 256, 256, 2, (3)),
    (8, 512, 16, 16, 4, (1.5)),
    (16, 1024, 23, 23, 7, (2.3)),
]


def Interpolate3d(shape, dtype, channels_last, backward):
    N, C, H, W, D, scale_factor = (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
    )

    if channels_last:
        input = (
            torch.randn(N, C, H, W, D, requires_grad=True)
            .to(memory_format=torch.channels_last_3d)
            .to(device=device, dtype=dtype)
        )
    else:
        input = torch.randn(N, C, H, W, D, requires_grad=True).to(
            device=device, dtype=dtype
        )

    output = torch.nn.functional.interpolate(
        input, scale_factor=shape[5], mode="nearest"
    )

    if backward:
        output.backward(torch.ones_like(output))


if __name__ == "__main__":
    backward = True
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                # warm up
                Interpolate3d(shape, dtype, channels_last, backward=True)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3], shape[4]),
                    "; datatype:",
                    dtype,
                    "; scale_factor:",
                    shape[5],
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
                        Interpolate3d(shape, dtype, channels_last, backward=True)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    Interpolate3d(shape, dtype, channels_last, backward=True)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
