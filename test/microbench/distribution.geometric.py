import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(8192, 8192)]
backward = False

if __name__ == "__main__":
    for shape in shape_list:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            input = torch.randn(shape, dtype=torch.bfloat16, device=torch.device("xpu"))

            # warm up
            input.geometric_(0.5)

            # go
            print(
                "shape:",
                (shape),
                "; datatype:",
                dtype,
                "; P:",
                0.5,
                "; backward:",
                backward,
            )
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
                record_shapes=True,
            ) as prof:
                for i in range(20):
                    input.geometric_(0.5)
            print(prof.key_averages().table(sort_by="xpu_time_total"))
