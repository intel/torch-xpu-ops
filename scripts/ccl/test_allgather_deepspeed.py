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

msg_size=150994944
dist.init_process_group("ccl")
rank = dist.get_rank()
size = dist.get_world_size()
device = "xpu:{}".format(rank)

torch.xpu.set_device(rank)

dst = torch.randn(msg_size * size).bfloat16().to("xpu:"+str(rank))
x = torch.randn(msg_size).bfloat16().to("xpu:"+str(rank))

dst2 = torch.randn(msg_size * size).bfloat16().to("xpu:"+str(rank))
x2 = torch.randn(msg_size).bfloat16().to("xpu:"+str(rank))

m = torch.nn.Linear(25600, 7680).bfloat16().to("xpu:"+str(rank))
m_in = torch.randn(16384, 25600).bfloat16().to("xpu:"+str(rank))

compute_stream = torch.xpu.current_stream()
tmp_stream = torch.xpu.Stream()
allgather_stream = torch.xpu.Stream()

print(compute_stream._as_parameter_)
print(allgather_stream._as_parameter_)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]) as prof:
    for i in range(5):
        # step 1: allgather for current sub module
        with torch.xpu.stream(allgather_stream):
            dist.all_gather_into_tensor(dst, x, async_op=False)
        for j in range(10):
            # step 2: wait stream
            compute_stream.wait_stream(allgather_stream)
            # step 3: prefech next allgather
            with torch.xpu.stream(allgather_stream):
                dist.all_gather_into_tensor(dst2, x2, async_op=False)
            # step 4: forward compute
            with torch.xpu.stream(compute_stream):
                out = m(m_in)
        torch.xpu.synchronize()

if rank == 0:
    prof.export_chrome_trace('./profile_kineto_trace_0.json')
if rank == 11:
    prof.export_chrome_trace('./profile_kineto_trace_11.json')
if rank == 23:
    prof.export_chrome_trace('./profile_kineto_trace_23.json')
print("DONE")

