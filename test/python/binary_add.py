import torch

a_cpu = torch.randn(2, 3)
b_cpu = torch.randn(2, 3)
a_xpu = a_cpu.to('xpu')
b_xpu = b_cpu.to('xpu')

c_xpu = a_xpu + b_xpu
c_cpu = a_cpu + b_cpu

print(torch.allclose(c_cpu, c_xpu.to('cpu')))
