import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import argparse
import os
import sys

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29800'

msg_size=1
dist.init_process_group("ccl")
rank = dist.get_rank()
size = dist.get_world_size()
device = "xpu:{}".format(rank)

torch.xpu.set_device(rank)

dst_list = [torch.randn(msg_size).bfloat16().to("xpu:"+str(rank)) for i in range(size)]
src = torch.randn(msg_size).bfloat16().to("xpu:"+str(rank))

m = torch.nn.Linear(16, 16).bfloat16().to("xpu:"+str(rank))
m_in = torch.randn(128, 16).bfloat16().to("xpu:"+str(rank))

for i in range(200):
    out = m(m_in)
    dist.all_gather(dst_list, src)

torch.xpu.synchronize()
print("DONE")

