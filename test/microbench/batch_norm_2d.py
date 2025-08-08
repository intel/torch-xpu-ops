import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
num_iter = 20
shape_list = [
    (256, 256, 56, 56, 256),
    (256, 2048, 7, 7, 2048),
    (24, 512, 28, 28, 512),
    (24, 1024, 14, 14, 1024),
    (4, 8, 640, 1024, 8),
    (4, 48, 20, 32, 48),
]


def BTN2d(shape, dtype, channels_last, backward):
    N, C, H, W, num_features = shape[0], shape[1], shape[2], shape[3], shape[4]

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
        grad = torch.randn([C, H, W]).to(device="xpu", dtype=dtype)

    BTN = torch.nn.BatchNorm2d(shape[4], device=device)

    output = BTN(input)

    if backward:
        output[0].backward(grad)


if __name__ == "__main__":
    backward = True
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for shape in shape_list:
            for channels_last in [False, True]:
                # warm up
                BTN2d(shape, dtype, channels_last, backward)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3]),
                    "; datatype:",
                    dtype,
                    "; num_features:",
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
                        BTN2d(shape, dtype, channels_last, backward=True)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    BTN2d(shape, dtype, channels_last, backward=True)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
