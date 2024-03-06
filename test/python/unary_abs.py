import torch

a_cpu = torch.randn(2, 3, 100)
a_xpu = a_cpu.to('xpu')

a_xpu.abs_()
c_cpu = a_cpu.abs()

print(torch.allclose(c_cpu, a_xpu.to('cpu')))
