import torch
import torch.distributed as dist
import argparse
import os
import sys

rank = 0
loop = 3
m1 = torch.nn.Linear(4, 4).to("xpu:"+str(rank))
m2 = torch.nn.Linear(4, 4).to("xpu:"+str(rank))
m_in = torch.randn(4, 4).to("xpu:"+str(rank))

current_stream = torch.xpu.current_stream()
unshard_stream_tmp = torch.xpu.Stream(torch.xpu.current_device())
unshard_stream = torch.xpu.Stream(torch.xpu.current_device())

out = torch.empty(4, 4).to("xpu:"+str(rank))

for i in range(loop):
    with torch.xpu.stream(current_stream):
        out = m1(m_in) + out
    unshard_stream.wait_stream(current_stream)
    with torch.xpu.stream(unshard_stream):
        out2 = m2(out) + out
    current_stream.wait_stream(unshard_stream)
    with torch.xpu.stream(current_stream):
        out3 = m1(out2) + out2
    unshard_stream.wait_stream(current_stream)
    with torch.xpu.stream(current_stream):
        out = m2(out3) + out3

print(f"first {out}")

#reference
out = torch.empty(4, 4).to("xpu:"+str(rank))

for i in range(loop):
    with torch.xpu.stream(current_stream):
      out = m1(m_in) + out
      out2 = m2(out) + out
      out3 = m1(out2) + out2
      out = m2(out3) + out3

print(f"second {out}")



