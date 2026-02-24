import torch
import torch.distributed as dist
import intel_extension_for_pytorch
import argparse
import os
import sys

# os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
# os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29800'


rank = 0
device = "xpu:{}".format(rank)

m_in = torch.randn(16384, 256).bfloat16().to(device)  #2560
m_out = torch.randn(16384, 7680).bfloat16().to(device)
m = torch.nn.Linear(256, 7680).bfloat16().to(device)

m_in_2 = torch.randn(16384, 256).bfloat16().to(device)
m_out_2 = torch.randn(16384, 7680).bfloat16().to(device)
m2 = torch.nn.Linear(256, 7680).bfloat16().to(device)

m_in_3 = torch.randn(16384, 256).bfloat16().to(device)
m_out_3 = torch.randn(16384, 7680).bfloat16().to(device)
m3 = torch.nn.Linear(256, 7680).bfloat16().to(device)

compute_stream = torch.xpu.current_stream()
compute_stream = torch.xpu.Stream(torch.xpu.current_device())
communication_stream = torch.xpu.Stream(torch.xpu.current_device())
compute_stream2 = torch.xpu.Stream(torch.xpu.current_device())

import contextlib
profiling = os.environ.get("PROFILE", "OFF").upper() in ["1", "Y", "ON", "YES", "TRUE"]
if profiling:
    prof = torch.profiler.profile()
else:
    prof = contextlib.nullcontext()

# sequence order
with prof:
    with torch.xpu.stream(compute_stream):
        for i in range(10):
            m_out = m_out + m(m_in)
    with torch.xpu.stream(communication_stream):
        for i in range(10):
            m_out_2 = m_out_2 + m2(m_in_2)
    with torch.xpu.stream(compute_stream2):
        for i in range(10):
            m_out_3 = m_out_3 + m3(m_in_3)
    torch.xpu.synchronize(device)

if profiling:
    torch.save(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1), 'profile_{}.pt'.format(rank))
    torch.save(prof.profiler.table(sort_by="id", row_limit=-1), 'profile_{}_id.pt'.format(rank))
    if hasattr(prof, "export_chrome_trace"):
        prof.export_chrome_trace('profile_trace.json')

