import torch
import torch.distributed as dist
import argparse
import os
import sys

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29800'

msg_size=20
dist.init_process_group("xccl")
rank = dist.get_rank()
size = dist.get_world_size()
device = "xpu:{}".format(rank)

dp_ranks = range(0, size)
dp_group = dist.new_group(dp_ranks)

dst = torch.randn(msg_size * size).bfloat16().to("xpu:"+str(rank))
x = torch.randn(msg_size).bfloat16().to("xpu:"+str(rank))

m = torch.nn.Linear(4, 4).bfloat16().to("xpu:"+str(rank))
m_in = torch.randn(64, 4).bfloat16().to("xpu:"+str(rank))

for i in range(100):
    dist.all_gather_into_tensor(dst, x, dp_group, async_op=False)

torch.xpu.synchronize()

print(f"DONE {dst}")
