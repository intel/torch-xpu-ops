import torch

def compute(input1, input2):
    input1 = input1.to(device='xpu')
    return input1 + 1.0

input1 = torch.randn(3,3,device='cpu')
input2 = torch.randn(3,3,device='cpu')

#warm
output = compute(input1, input2)

for id in range(1):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.XPU,]) as p:
        output = compute(input1, input2)
    print(p.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1))
