import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import argparse
import os
import sys
import contextlib

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

m = torch.nn.Linear(2560, 7680).bfloat16().to("xpu:"+str(rank))
m_in = torch.randn(16384, 2560).bfloat16().to("xpu:"+str(rank))

print("--kineto profiling--")
os.environ["IPEX_ZE_TRACING"] = "1"
os.environ["ZE_ENABLE_TRACING_LAYER"] = "1"
print(os.environ["IPEX_ZE_TRACING"])
print(os.environ["ZE_ENABLE_TRACING_LAYER"])

#warm up
for i in range(5):
    out = m(m_in)
    dist.all_gather_into_tensor(dst, x, async_op=False)
    out_2 = out + m(m_in)
    dist.all_gather_into_tensor(dst, x, async_op=False)
    out_3 = out_2 + m(m_in)
    dist.all_gather_into_tensor(dst, x, async_op=False)
    out_4 = out_3 + m(m_in)
    dist.all_gather_into_tensor(dst, x, async_op=False)
    out_5 = out_3 + m(m_in)
    dist.all_gather_into_tensor(dst, x, async_op=False)
    out_6 = out_5 + m(m_in)
    dist.all_gather_into_tensor(dst, x, async_op=False)

torch.xpu.synchronize()
'''
# cases define
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]) as prof:
    for i in range(10):
        out = m(m_in)
        dist.all_gather_into_tensor(dst, x, async_op=False)
        out_2 = out + m(m_in)
        dist.all_gather_into_tensor(dst, x, async_op=False)
        out_3 = out_2 + m(m_in)
        dist.all_gather_into_tensor(dst, x, async_op=False)
        out_4 = out_3 + m(m_in)
        dist.all_gather_into_tensor(dst, x, async_op=False)
        out_5 = out_3 + m(m_in)
        dist.all_gather_into_tensor(dst, x, async_op=False)
        out_6 = out_5 + m(m_in)
        dist.all_gather_into_tensor(dst, x, async_op=False)
        torch.xpu.synchronize()

torch.save(prof.key_averages().table(sort_by="self_xpu_time_total"), 'profile_kineto_{}.pt'.format(rank))
torch.save(prof.key_averages(group_by_input_shape=True).table(), "./profile_kineto_{}_detail.pt".format(rank))
if rank == 0:
    prof.export_chrome_trace('./profile_kineto_trace.json')
'''
