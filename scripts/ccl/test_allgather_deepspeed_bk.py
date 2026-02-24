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

dst = torch.randn(msg_size * size).bfloat16().to("xpu:"+str(rank))
x = torch.randn(msg_size).bfloat16().to("xpu:"+str(rank))

dst2 = torch.randn(msg_size * size).bfloat16().to("xpu:"+str(rank))
x2 = torch.randn(msg_size).bfloat16().to("xpu:"+str(rank))

m = torch.nn.Linear(2560, 7680).bfloat16().to("xpu:"+str(rank))
m_in = torch.randn(16384, 2560).bfloat16().to("xpu:"+str(rank))

compute_stream = torch.xpu.current_stream()
allgather_stream = torch.xpu.Stream(torch.xpu.current_device())

for i in range(5):
    for j in range(2):
        # step 1: allgather for current sub module
        with torch.xpu.stream(allgather_stream):
          dist.all_gather_into_tensor(dst, x, async_op=False)
        # step 2: wait stream
        compute_stream.wait_stream(allgather_stream)
        # step 3: prefect next allgather
        with torch.xpu.stream(allgather_stream):
          dist.all_gather_into_tensor(dst2, x2, async_op=False)
        # step 4: forward compute
        with torch.xpu.stream(compute_stream):
          out = m(m_in)
    torch.xpu.synchronize()

print("DONE")