import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
num_iter = 20

shape_list = [
    (8, 32, 256, 256, (3)),
    (8, 512, 16, 16, (1.5)),
    (16, 1024, 23, 23, (2.3)),
    (4, 32, 80, 128, (2)),
]


def Interpolate2d(shape, dtype, channels_last, backward, mode):
    N, C, H, W, scale_factor = shape[0], shape[1], shape[2], shape[3], shape[4]

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

    output = torch.nn.functional.interpolate(input, scale_factor=shape[4], mode=mode)

    if backward:
        output.backward(torch.ones_like(output))


if __name__ == "__main__":
    backward = True
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                for mode in ["nearest-exact"]:
                    # warm up
                    Interpolate2d(shape, dtype, channels_last, backward, mode)

                    # go
                    print(
                        "shape:",
                        (shape[0], shape[1], shape[2], shape[3]),
                        "; datatype:",
                        dtype,
                        "; scale_factor:",
                        shape[4],
                        "; mode:",
                        mode,
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
                            Interpolate2d(shape, dtype, channels_last, backward, mode)
                    print(prof.key_averages().table(sort_by="xpu_time_total"))

                    # E2E time
                    torch.xpu.synchronize()
                    t1 = time.time()
                    for i in range(num_iter):
                        Interpolate2d(shape, dtype, channels_last, backward, mode)
                    torch.xpu.synchronize()
                    t2 = time.time()
                    e2e_time = (t2 - t1) / num_iter
                    print("E2E total time:", f"{float(e2e_time):.20f}")
