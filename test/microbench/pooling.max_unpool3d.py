import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (2, 32, 64, 64, 64),
    (4, 33, 64, 64, 64),
    (16, 32, 32, 32, 32),
]


def maxUnpool3d(shape, dtype, device, channels_last, backward):
    N, C, D, H, W = (
        int(shape[0]),
        int(shape[1]),
        int(shape[2]),
        int(shape[3]),
        int(shape[4]),
    )
    kernel_size = 2

    pool = torch.nn.MaxPool3d(kernel_size, return_indices=True).to(
        device=device, dtype=dtype
    )
    unpool = torch.nn.MaxUnpool3d(kernel_size).to(device=device, dtype=dtype)
    torch.manual_seed(20)

    if channels_last:
        input = (
            torch.randn([N, C, D, H, W])
            .to(memory_format=torch.channels_last_3d)
            .to(device=device, dtype=torch.float32)
        )
    else:
        input = torch.randn([N, C, D, H, W]).to(device=device, dtype=torch.float32)
    output, indices = pool(input)

    if channels_last:
        x_dpcpp = output.to(memory_format=torch.channels_last_3d).to(
            device=device, dtype=dtype
        )
        indices_dpcpp = indices.to(memory_format=torch.channels_last_3d).to(
            device=device, dtype=torch.int64
        )
    else:
        x_dpcpp = output.to(device=device, dtype=dtype)
        indices_dpcpp = indices.to(device=device, dtype=torch.int64)

    if backward:
        x_dpcpp.requires_grad_(True)
        if channels_last:
            grad_dpcpp = (
                torch.randn([N, C, D, H, W])
                .to(memory_format=torch.channels_last_3d)
                .to(device=device, dtype=dtype)
            )
        else:
            grad_dpcpp = torch.randn([N, C, D, H, W]).to(device=device, dtype=dtype)

    y_dpcpp = unpool(x_dpcpp, indices_dpcpp, output_size=torch.Size([N, C, D, H, W]))

    if backward:
        y_dpcpp.backward(grad_dpcpp)


if __name__ == "__main__":
    backward = True
    device = "xpu"
    num_iter = 20
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                # warm up
                maxUnpool3d(shape, dtype, device, channels_last, backward=backward)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3], shape[4]),
                    "; datatype:",
                    dtype,
                    "; kernel_size:",
                    str(2),
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
                        maxUnpool3d(
                            shape, dtype, device, channels_last, backward=backward
                        )
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    maxUnpool3d(shape, dtype, device, channels_last, backward=backward)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
