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

msg_size=204800
dist.init_process_group("ccl")
rank = dist.get_rank()
size = dist.get_world_size()
device = "xpu:{}".format(rank)

torch.xpu.set_device(rank)

dst = torch.randn(msg_size * size).bfloat16().to(device)
x = torch.randn(msg_size).bfloat16().to(device)

m_in = torch.randn(2048, 256).bfloat16().to(device)
m = torch.nn.Linear(256, 7680).bfloat16().to(device)

#step1 warmup
out = m(m_in)
dist.all_reduce(out)
x2 = x + 1
dist.all_reduce(x2)
x3 = x2 + 2
dist.all_reduce(x3)
torch.xpu.synchronize()

# record
g = torch.xpu.XPUGraph()
with torch.xpu.graph(g):
    out = m(m_in)
    dist.all_reduce(out)
    x2 = x + 1
    dist.all_reduce(x2)
    x3 = x2 + 2
    dist.all_reduce(x3)
    torch.xpu.synchronize()

# replay
for i in range(2):
    g.replay()
torch.xpu.synchronize()

print("DONE")
