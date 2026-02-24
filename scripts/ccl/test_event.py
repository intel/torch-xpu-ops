
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

msg_size=2048
dist.init_process_group("ccl")
rank = dist.get_rank()
size = dist.get_world_size()
device = "xpu:{}".format(rank)

torch.xpu.set_device(device)

tmp_stream = torch.Stream()

tmp_event = torch.Event()
tmp_event.record()

print(f"zl_debug event device {tmp_event.device}")
print(f"zl_debug stream device {tmp_stream.device}")

tmp_stream.wait_event(tmp_event)