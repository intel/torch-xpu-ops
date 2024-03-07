import torch

a_cpu = torch.randn(2, 3, 4)
a_xpu = a_cpu.to('xpu')

a_xpu.resize_(4, 3, 2)
b_xpu = torch.full_like(a_xpu, 1)
c_cpu = torch.ones([4, 3, 2])

assert b_xpu.shape[0] == 4
assert b_xpu.shape[1] == 3
assert b_xpu.shape[2] == 2

print(torch.allclose(c_cpu, b_xpu.to('cpu')))
