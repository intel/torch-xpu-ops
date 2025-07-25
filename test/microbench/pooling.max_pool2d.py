import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (16, 24, 112, 112, (3), (2)),
    (16, 1984, 7, 7, (3, 2), (2, 1)),
    (64, 1024, 112, 112, (6), (4)),
    (16, 2048, 224, 224, (3), (2)),
]


def mp2d(shape, dtype, channels_last, backward):
    N, C, H, W, kernel_size, stride = (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
    )

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
        if isinstance(kernel_size, int):
            Wout = (W - kernel_size) / stride + 1
            Hout = (H - kernel_size) / stride + 1
        else:
            Wout = (W - kernel_size[1]) / stride[1] + 1
            Hout = (H - kernel_size[0]) / stride[0] + 1
        grad = torch.randn([N, C, int(Hout), int(Wout)]).to(device="xpu", dtype=dtype)

    mp2d = torch.nn.MaxPool2d(shape[4], stride=shape[5], return_indices=True)

    output = mp2d(input)

    if backward:
        output[0].backward(grad)


if __name__ == "__main__":
    backward = True
    num_iter = 20
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                # warm up
                mp2d(shape, dtype, channels_last, backward)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3]),
                    "; datatype:",
                    dtype,
                    "; kernel_size:",
                    shape[4],
                    "; stride:",
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
                        mp2d(shape, dtype, channels_last, backward)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    mp2d(shape, dtype, channels_last, backward)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
