import time

import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (16, 32, 64, 64, 64, 32, 32, 32),
    (1, 4, 144, 144, 144, 72, 72, 72),
    (512, 512, 12, 12, 12, 6, 6, 6),
]
num_iter = 20


def fmp3d(shape, dtype, channels_last, backward):
    torch.manual_seed(20)
    N, C, H, W, D, oH, oW, oD = (
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        shape[4],
        shape[5],
        shape[6],
        shape[7],
    )

    if channels_last:
        input = (
            torch.randn(N, C, H, W, D)
            .to(memory_format=torch.channels_last_3d)
            .to(device="xpu", dtype=dtype)
        )
    else:
        input = torch.randn(N, C, H, W, D).to(device="xpu", dtype=dtype)

    if backward:
        input.requires_grad_(True)
        grad = torch.randn([N, C, oH, oW, oD]).to(device="xpu", dtype=dtype)

    fmp = torch.nn.MaxPool3d(2, return_indices=True)
    output = fmp(input)

    # warm up
    output = fmp(input)
    if backward:
        output[0].backward(grad)

    # go
    print(
        "shape:",
        (shape[0], shape[1], shape[2], shape[3], shape[4]),
        "; datatype:",
        dtype,
        "; channels_last:",
        channels_last,
        "; backward:",
        backward,
    )
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
    ) as prof:
        for i in range(num_iter):
            output = fmp(input)
            if backward:
                output[0].backward(grad)
    print(prof.key_averages().table(sort_by="xpu_time_total"))

    # E2E time
    torch.xpu.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        output = fmp(input)
        if backward:
            output[0].backward(grad)
    torch.xpu.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


if __name__ == "__main__":
    backward = True
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            for channels_last in [False, True]:
                fmp3d(shape, dtype, channels_last, backward=True)
