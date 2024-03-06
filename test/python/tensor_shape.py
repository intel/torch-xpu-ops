import torch

a_cpu = torch.randn(2, 3, 4)
a_xpu = a_cpu.to('xpu')

a_xpu = a_xpu.view(4, 3, 2)

assert a_xpu.shape[0] == 4
assert a_xpu.shape[1] == 3
assert a_xpu.shape[2] == 2
