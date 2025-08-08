import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
num_iter = 20
shape_list = [(2, 5, 6, 3, 5, 5), (2, 8, 64, 64, 64, 8), (16, 16, 128, 128, 256, 16)]


def BTN3d(shape, dtype, channels_last, backward):
    N, C, D, H, W, num_features = (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
    )

    if channels_last:
        input = (
            torch.randn(N, C, D, H, W)
            .to(memory_format=torch.channels_last_3d)
            .to(device="xpu", dtype=dtype)
        )
    else:
        input = torch.randn(N, C, D, H, W).to(device="xpu", dtype=dtype)

    if backward:
        input.requires_grad_(True)
        grad = torch.randn([C, D, H, W]).to(device="xpu", dtype=dtype)

    BTN = torch.nn.BatchNorm3d(shape[5], device=device)

    output = BTN(input)

    if backward:
        output[0].backward(grad)


if __name__ == "__main__":
    backward = True
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for shape in shape_list:
            for channels_last in [False, True]:
                # warm up
                BTN3d(shape, dtype, channels_last, backward)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3], shape[4]),
                    "; datatype:",
                    dtype,
                    "; num_features:",
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
                        BTN3d(shape, dtype, channels_last, backward=True)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    BTN3d(shape, dtype, channels_last, backward=True)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
