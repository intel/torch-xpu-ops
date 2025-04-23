import torch
from torch.profiler import profile, ProfilerActivity

shape_list = [(1024, 8)]
device = "xpu"
backward = True
dict_len = 2500000
vect_len = 128

for shape in shape_list:
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        emb = torch.nn.Embedding(dict_len, vect_len, dtype=dtype, device=device)
        input = torch.randint(0, dict_len, (1024, 8), device=device)
        grad = torch.randn(1024, 8, vect_len, dtype=dtype, device=device)

        # warm up
        output = emb(input)
        output.backward(grad)

        # go
        print("shape:", (shape), "; datatype:", dtype, "; backward:", backward)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for i in range(20):
                output = emb(input)
                output.backward(grad)
        print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=100))
