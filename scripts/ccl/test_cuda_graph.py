
import torch

m_in = torch.randn(16384, 2560).bfloat16().to("cuda")
m1 = torch.nn.Linear(2560, 7680).bfloat16().to("cuda")
m2 = torch.nn.Linear(2560, 768).bfloat16().to("cuda")

s1 = torch.cuda.current_stream()
s2 = torch.cuda.Stream()

# warmup
s2.wait_stream(s1)
for i in range(3):
    with torch.cuda.stream(s1):
        out1 = m1(m_in)
        out1_1 = out1 * 3
        out1_2 = out1_1 + 10.0
    with torch.cuda.stream(s2):
        out2 = m2(m_in)
        out2_1 = out2 * 3
        out2_2 = out2_1 + 10.0
s1.wait_stream(s2)

# capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    with torch.cuda.stream(s1):
        out1 = m1(m_in)
        out1_1 = out1 * 3
        out1_2 = out1_1 + 10.0
    with torch.cuda.stream(s2):
        out2 = m2(m_in)
        out2_1 = out2 * 3
        out2_2 = out2_1 + 10.0

real_inputs = [torch.rand_like(m_in) for _ in range(10)]

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU]) as prof:
    for data in real_inputs:
        # Fills the graph's input memory with new data to compute on
        m_in.copy_(data)
        g.replay()

prof.export_chrome_trace('./profile_kineto_trace.json')
