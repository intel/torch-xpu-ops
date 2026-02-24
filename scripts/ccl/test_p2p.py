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

value = rank + 1
in_tensor = torch.empty(value, value, value, dtype=torch.float32).fill_(value).to(device)

# rank0 -> rank1
src = 0
dst = 1
if rank == src:
    # Send
    dist.send(in_tensor, dst)
elif rank == dst:
    # Recv
    output_tensor = torch.empty(src+1, src+1, src+1, dtype=torch.float32).fill_(-1).to(device)
    dist.recv(output_tensor, src)

torch.xpu.synchronize()+
print("done")
