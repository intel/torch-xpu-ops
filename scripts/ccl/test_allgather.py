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

msg_size=19669120
dist.init_process_group("ccl")
rank = dist.get_rank()
size = dist.get_world_size()
device = "xpu:{}".format(rank)

dp_ranks = range(0, size)
dp_group = dist.new_group(dp_ranks)

dst = torch.randn(msg_size * size).bfloa

t16().to("xpu:"+str(rank))
x = torch.randn(msg_size).bfloat16().to("xpu:"+str(rank))

m1 = torch.nn.Linear(4, 4).to("xpu:"+str(rank))
m2 = torch.nn.Linear(4, 5).to("xpu:"+str(rank))
m_in = torch.randn(64, 4).to("xpu:"+str(rank))

current_stream = torch.xpu.current_stream()
unshard_stream_tmp = torch.xpu.Stream(torch.xpu.current_device())
unshard_stream = torch.xpu.Stream(torch.xpu.current_device())

for i in range(5):
    with torch.xpu.stream(current_stream):
      out = m1(m_in)
    with torch.xpu.stream(unshard_stream):
      dist.all_gather_into_tensor(dst, out)
    with torch.xpu.stream(current_stream):
      out2 = m2(dst)
    with torch.xpu.stream(unshard_stream):
      out3 = dist.all_gather_into_tensor(dst2, out2)

torch.xpu.synchronize()
