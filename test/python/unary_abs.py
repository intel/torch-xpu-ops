import torch

a_cpu = torch.randn(2, 3)
a_xpu = a_cpu.to('xpu')

c_xpu = a_xpu.abs()
c_cpu = a_cpu.abs()

print(torch.allclose(c_cpu, c_xpu.to('cpu')))
