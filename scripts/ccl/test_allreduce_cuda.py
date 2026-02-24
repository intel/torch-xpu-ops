import torch
import torch.distributed as dist
import argparse
import os
import sys

# os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
# os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29800'

msg_size=19669120
dist.init_process_group("cuda")
rank = dist.get_rank()
size = dist.get_world_size()
device = "cuda:{}".format(rank)

dp_ranks = range(0, size)
dp_group = dist.new_group(dp_ranks)

dst = torch.randn(msg_size * size).bfloat16().to(device)
x = torch.randn(msg_size).bfloat16().to(device)

m_in = torch.randn(16384, 2560).bfloat16().to(device)
m = torch.nn.Linear(2560, 7680).bfloat16().to(device)
m2 = torch.nn.Linear(2560, 7680).bfloat16().to(device)
m3 = torch.nn.Linear(2560, 7680).bfloat16().to(device)
m4 = torch.nn.Linear(2560, 7680).bfloat16().to(device)
m5 = torch.nn.Linear(2560, 7680).bfloat16().to(device)
m6 = torch.nn.Linear(2560, 7680).bfloat16().to(device)

allreduce_in_fp32 = torch.randn(16384, 2560).to(device)

compute_stream = torch.cuda.stream()
communication_stream = torch.cuda.stream()

for i in range(5):
    with torch.xpu.stream(compute_stream):
      out = m(m_in)
      out2 = out + m2(m_in)
      out3 = out + m3(m_in)
      out4 = out + m4(m_in)
      out5 = out + m5(m_in)
      out6 = out + m6(m_in)

    with torch.xpu.stream(communication_stream):
      # cast to torch bf16
      x = torch.cast_to_bf16(allreduce_in_fp32)
      dist.all_reduce(x, group=dp_group)
      x2 = x + 1
      dist.all_reduce(x2, group=dp_group)
      x3 = x2 + 2
      dist.all_reduce(x3, group=dp_group)

torch.cuda.synchronize()
