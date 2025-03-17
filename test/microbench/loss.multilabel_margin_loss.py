import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
shape_list = [
    (8192, 8192)
]

cache_r = torch.randn((1024 * 1024 * 1024), device="xpu")
cache_w = torch.randn((1024 * 1024 * 1024), device="xpu")

def _test_dpcpp(input, target, reduce, dtype):
    loss = nn.MultiLabelMarginLoss(reduction=reduce)
    input.requires_grad = True

    if(reduce == "none"):
        # warm up
        output = loss(input, target)
        output.backward(torch.ones_like(output, dtype=dtype).to("xpu"))

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward, "; reduce: 0" if(reduce == "none") else "; reduce: 1")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:    
            for i in range(20):
                cache_r = cache_w
                output = loss(input, target)
                cache_r = cache_w
                output.backward(torch.ones_like(output, dtype=dtype).to("xpu"))
        print(prof.key_averages().table(sort_by="xpu_time_total"))
        
    else:
        # warm up
        output = loss(input, target)
        output.backward(torch.tensor((1.0), dtype=dtype).to("xpu"))
        
        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward, "; reduce: 0" if(reduce == "none") else "; reduce: 1")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True) as prof:
            for i in range(20):
                cache_r = cache_w
                output = loss(input, target)
                cache_r = cache_w
                output.backward(torch.tensor((1.0), dtype=dtype).to("xpu"))
        print(prof.key_averages().table(sort_by="xpu_time_total")) 

for shape in shape_list:
    for reduce in ["none", "mean"]:
        for dtype in [torch.bfloat16, torch.float16, torch.float32]:
            input = torch.randn(shape, dtype=dtype)
            target = torch.randn(shape, dtype=dtype).long()
            input_dpcpp = input.to("xpu")
            target_dpcpp = target.to("xpu")
            _test_dpcpp(input_dpcpp, target_dpcpp, reduce, dtype)
