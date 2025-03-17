import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [
              
              (6, 5, 7, 3),
              (4, 32, 128, 128),
              (16, 128, 1024, 1024),
            ]

device = "xpu"
backward = True

for shape in shape_list:
    for mode in ['bilinear', 'nearest', 'bicubic']:
        for padding_mode in ['zeros', 'border', 'reflection']:
            for align_corners in [True, False]:
                for dtype in [torch.bfloat16, torch.float16, torch.float32]:
                    N, C, H, W = shape

                    input = torch.randn(N, C, H, W, dtype=dtype, device=device)
                    grid = torch.randn(N, H, W, 2, dtype=dtype, device=device)
                    
                    if backward:
                        input.requires_grad_(True)
                        grid.requires_grad_(True)
                    
                    # warm up
                    output = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,align_corners=align_corners)
                    if backward:
                        output.sum().backward()

                    # go
                    print("shape:", (shape), "; datatype:", dtype, "; mode:", mode, "; padding_mode:", padding_mode, "; align_corners:", align_corners, "; backward:", backward)
                    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:
                        for i in range(20):
                            output = torch.nn.functional.grid_sample(input, grid, mode=mode, padding_mode=padding_mode,
                                    align_corners=align_corners)
                            if backward:
                                output.sum().backward()
                    print(prof.key_averages().table(sort_by="xpu_time_total"))
