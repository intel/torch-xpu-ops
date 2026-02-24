import torch
import intel_extension_for_pytorch
import torch.distributed as dist
import argparse
import os
import sys


os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
tmp_rank = os.environ['RANK']
print(tmp_rank)

rank0 = 0
device0 = "xpu:{}".format(rank0)
rank1 = 1
device1 = "xpu:{}".format(rank1)

m_in_0 = torch.randn(2048,4096).half()
m_in_1 = torch.randn(2048,4096).half()
dst_0 = torch.randn(2048,4096).half().to(device0)
dst_1 = torch.randn(2048,4096).half().to(device1)

for i in range(1000):
    m_tmp = m_in_0.to(device0)
    m_tmp.copy_(dst_0)
    m_tmp_1 = m_in_1.to(device1)
    m_tmp_1.copy_(dst_1)

torch.xpu.synchronize(device0)
torch.xpu.synchronize(device1)

