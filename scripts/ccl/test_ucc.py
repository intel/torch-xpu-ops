
import torch
import torch.distributed as dist
import argparse
import os
import sys

# os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
# os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29800'

dist.init_process_group("ucc")
rank = dist.get_rank()
size = dist.get_world_size()

allreduce_in_fp32 = torch.randn(16384, 2560)

# cast to torch bf16
x = torch.cast_to_bf16(allreduce_in_fp32)
dist.all_reduce(x)
x2 = x + 1
dist.all_reduce(x2)
x3 = x2 + 2
dist.all_reduce(x3)
