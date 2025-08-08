import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"

shape_list = [
    (16, 24, 28, 19, 19, (3), (2)),
    (16, 1984, 7, 7, 7, (3, 2, 2), (2, 1, 2)),
    (64, 1024, 14, 14, 14, (6), (4)),
]
num_iter = 20


def AVGPool3d(shape, dtype, channels_last, backward):
    N, C, D, H, W, kernel_size, stride = (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
        shape[6],
    )

    if channels_last:
        input = (
            torch.randn(N, C, D, H, W)
            .to(memory_format=torch.channels_last_3d)
            .to(device=device, dtype=dtype)
        )
    else:
        input = torch.randn(N, C, D, H, W).to(device=device, dtype=dtype)

    if backward:
        input.requires_grad_(True)
        if isinstance(kernel_size, int):
            Dout = (D - kernel_size) / stride + 1
            Hout = (H - kernel_size) / stride + 1
            Wout = (W - kernel_size) / stride + 1
        else:
            Dout = (D - kernel_size[0]) / stride[0] + 1
            Hout = (H - kernel_size[1]) / stride[1] + 1
            Wout = (W - kernel_size[2]) / stride[2] + 1
        grad = torch.randn([C, int(Dout), int(Hout), int(Wout)]).to(
            device=device, dtype=dtype
        )

    AVG3d = torch.nn.AvgPool3d(shape[5], stride=shape[6])

    output = AVG3d(input)

    if backward:
        output[0].backward(grad)


if __name__ == "__main__":
    backward = True
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                # warm up
                AVGPool3d(shape, dtype, channels_last, backward)

                # go
                print(
                    "shape:",
                    (shape[0], shape[1], shape[2], shape[3], shape[4]),
                    "; datatype:",
                    dtype,
                    "; kernel_size:",
                    shape[5],
                    "; stride:",
                    shape[6],
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
                        AVGPool3d(shape, dtype, channels_last, backward=True)
                print(prof.key_averages().table(sort_by="xpu_time_total"))

                # E2E time
                torch.xpu.synchronize()
                t1 = time.time()
                for i in range(num_iter):
                    AVGPool3d(shape, dtype, channels_last, backward=True)
                torch.xpu.synchronize()
                t2 = time.time()
                e2e_time = (t2 - t1) / num_iter
                print("E2E total time:", f"{float(e2e_time):.20f}")
