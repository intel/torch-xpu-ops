import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
shape_list = [(8192, 8192)]
backward = False

for shape in shape_list:
    for replacement in [False, True]:
        for num_samples in [2, 128]:
            for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                input = torch.randn(shape, dtype=dtype, device=device).abs()
                # warm up
                input.multinomial(num_samples, replacement)

                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; replacement:",
                    replacement,
                    "; num_samples:",
                    num_samples,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[
                        ProfilerActivity.CPU,
                        ProfilerActivity.XPU,
                    ],
                    record_shapes=True,
                ) as prof:
                    for _ in range(20):
                        input.multinomial(num_samples, replacement)
                print(prof.key_averages().table(sort_by="xpu_time_total"))
