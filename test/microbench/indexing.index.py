import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(4, 15000)]
device = "xpu"
backward = False

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        for mode in ["with_nonzero", "without_nonzero"]:
            d = torch.rand(shape, dtype=dtype, device=device)
            e = torch.rand(shape, dtype=dtype, device=device)

            if mode == "with_nonzero":
                # warm up
                for i in range(100):
                    f = d < e
                    g = e[f]

                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; mode:",
                    mode,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(20):
                        f = d < e
                        g = e[f]
                print(prof.key_averages().table(sort_by="xpu_time_total"))
            else:
                f = torch.linspace(0, 4 - 2, steps=int(4 / 2), device=device).to(
                    torch.long
                )
                # warm up
                for i in range(100):
                    g = e[f]

                # go
                print(
                    "shape:",
                    (shape),
                    "; datatype:",
                    dtype,
                    "; mode:",
                    mode,
                    "; backward:",
                    backward,
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                    record_shapes=True,
                ) as prof:
                    for i in range(20):
                        g = e[f]
                print(prof.key_averages().table(sort_by="xpu_time_total"))
