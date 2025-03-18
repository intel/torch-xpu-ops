import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
    (32, 32, 128, 128, 128, 64, 64, 64),
    (1, 3, 144, 144, 144, 72, 72, 72),
    (512, 512, 12, 12, 12, 6, 6, 6),
]


def fmp3d(shape, dtype, channels_last, backward):
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

    fmp = torch.nn.FractionalMaxPool3d(2, output_size=(oH, oW, oD), return_indices=True)

    output = fmp(input)

    if backward:
        output[0].backward(grad)


if __name__ == "__main__":
    backward = True
    for channels_last in [False, True]:
        for shape in shape_list:
            for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                # warm up
                fmp3d(shape, dtype, channels_last, backward)

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
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(20):
                        fmp3d(shape, dtype, channels_last, backward)
                print(prof.key_averages().table(sort_by="xpu_time_total"))
