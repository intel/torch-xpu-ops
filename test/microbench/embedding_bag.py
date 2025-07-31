import random
import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
num_iter = 20

for dtype in [torch.bfloat16, torch.float16, torch.float32]:
    for reduce in ["max", "mean", "sum"]:
        dict_len = 2500000
        vect_len = 128
        batch = 1024

        emb = torch.nn.EmbeddingBag(
            dict_len, vect_len, mode=reduce, dtype=dtype, device=device
        )
        input = torch.empty([batch], dtype=torch.long, device=device)
        for i in range(batch):
            input[i] = random.randint(0, dict_len - 1)

        bag = torch.empty([batch], dtype=torch.long, device=device)
        for i in range(batch):
            bag[i] = i

        if backward:
            grad = torch.randn(batch, vect_len, dtype=dtype, device=device)

        # warm up
        for i in range(5):
            output = emb(input, bag)
            if backward:
                output.backward(grad)

        # go
        print(
            "shape:",
            (batch),
            "; datatype:",
            dtype,
            "; reduce:",
            reduce,
            "; backward:",
            backward,
        )
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU],
            record_shapes=True,
        ) as prof:
            for i in range(20):
                output = emb(input, bag)
                if backward:
                    output.backward(grad)
        print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=100))

        # E2E time
        torch.xpu.synchronize()
        t1 = time.time()
        for i in range(num_iter):
            output = emb(input, bag)
            if backward:
                output.backward(grad)
        torch.xpu.synchronize()
        t2 = time.time()
        e2e_time = (t2 - t1) / num_iter
        print("E2E total time:", f"{float(e2e_time):.20f}")
