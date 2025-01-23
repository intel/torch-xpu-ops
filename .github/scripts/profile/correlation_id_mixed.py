import torch

from torch.profiler import profile, record_function, ProfilerActivity

input1 = torch.randn(3, 3, device='xpu')
input2 = torch.randn(3, 3, device='xpu')

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU],) as prof:
    output1 = input1 + 1.0
    output2 = input2 + 2.0
    output = output1 + output2
print(prof.key_averages().table(sort_by="xpu_time_total"))
