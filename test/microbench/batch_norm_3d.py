import torch
from torch.profiler import profile, ProfilerActivity


if torch.cuda.is_available():
    device = "cuda"
    activity = ProfilerActivity.CUDA
    table_key = "cuda_time_total"
else:
    device = "xpu"
    activity = ProfilerActivity.XPU
    table_key = "xpu_time_total"


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
            .to(device=device, dtype=dtype)
        )
    else:
        input = torch.randn(N, C, D, H, W).to(device=device, dtype=dtype)

    if backward:
        input.requires_grad_(True)
        grad = torch.randn([C, D, H, W]).to(device=device, dtype=dtype)

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
                    activities=[ProfilerActivity.CPU, activity],
                    record_shapes=True,
                ) as prof:
                    for i in range(20):
                        BTN3d(shape, dtype, channels_last, backward=True)
                print(prof.key_averages().table(sort_by=table_key))
