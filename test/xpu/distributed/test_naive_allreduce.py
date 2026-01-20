import torch
import torch.distributed as dist
import argparse
import os
import sys

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29800'

msg_size=19669120
dist.init_process_group("xccl")
rank = dist.get_rank()
size = dist.get_world_size()
device = "xpu:{}".format(rank)
torch.xpu.set_device(device)


dp_ranks = range(0, size)
dp_group = dist.new_group(dp_ranks)

dst = torch.randn(msg_size * size).bfloat16().to("xpu:"+str(rank))
x = torch.randn(msg_size).bfloat16().to("xpu:"+str(rank))

m1 = torch.nn.Linear(4, 4).to("xpu:"+str(rank))
m2 = torch.nn.Linear(4, 5).to("xpu:"+str(rank))
m_in = torch.randn(64, 4).to("xpu:"+str(rank))

for i in range(5):
    dist.all_reduce(dst)

torch.xpu.synchronize()

print("!!!!!!!!!!!! DONE \n")

