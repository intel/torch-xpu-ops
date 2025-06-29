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


shape_list = [((64, 8), (8)), ((4, 128, 15000), (128)), ((4, 256, 512), (256))]

for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    for shape in shape_list:
        backward = True
        # input
        input = torch.randn(shape[0], device=device, dtype=dtype)

        if backward:
            input.requires_grad_(True)

        # warm up
        m = torch.nn.BatchNorm1d(shape[1], device=device)
        output = m(input)

        print(
            "shape:",
            shape[0],
            "; datatype:",
            dtype,
            "; num_features:",
            shape[1],
            "; backward:",
            backward,
        )
        with profile(
            activities=[ProfilerActivity.CPU, activity], record_shapes=True
        ) as prof:
            for i in range(20):
                m = torch.nn.BatchNorm1d(shape[1], device=device)
                output = m(input)
                if backward:
                    gy = torch.empty_like(output)
                    output.backward(gy)
        print(prof.key_averages().table(sort_by=table_key))
