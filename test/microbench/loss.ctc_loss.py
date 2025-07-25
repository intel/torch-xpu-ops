import time

import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True
num_iter = 20
# T,N,C,S
shape_list = [(32, 32, 32, 16), (128, 128, 128, 128), (8, 8, 4, 8)]


def _test_loss_ctc(log_probs, targets, input_lengths, target_lengths, dtype):
    log_probs_dpcpp = log_probs.to("xpu")
    log_probs_dpcpp.requires_grad_(True)
    targets_dpcpp = targets.to("xpu")
    input_lengths_dpcpp = input_lengths.to("xpu")
    target_lengths_dpcpp = target_lengths.to("xpu")

    # warm up
    loss_dpcpp = torch.nn.functional.ctc_loss(
        log_probs_dpcpp, targets_dpcpp, input_lengths_dpcpp, target_lengths_dpcpp
    )
    loss_dpcpp.backward()

    # go
    print(
        "shape:",
        (shape[0], shape[1], shape[2], shape[3]),
        "; datatype:",
        dtype,
        "; backward:",
        backward,
    )
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
    ) as prof:
        for i in range(num_iter):
            loss_dpcpp = torch.nn.functional.ctc_loss(
                log_probs_dpcpp,
                targets_dpcpp,
                input_lengths_dpcpp,
                target_lengths_dpcpp,
            )
            loss_dpcpp.backward()
    print(prof.key_averages().table(sort_by="xpu_time_total"))

    # E2E time
    torch.xpu.synchronize()
    t1 = time.time()
    for i in range(num_iter):
        loss_dpcpp = torch.nn.functional.ctc_loss(
            log_probs_dpcpp,
            targets_dpcpp,
            input_lengths_dpcpp,
            target_lengths_dpcpp,
        )
        loss_dpcpp.backward()
    torch.xpu.synchronize()
    t2 = time.time()
    e2e_time = (t2 - t1) / num_iter
    print("E2E total time:", f"{float(e2e_time):.20f}")


for shape in shape_list:
    for dtype in [torch.float32]:
        T, N, C, S = shape[0], shape[1], shape[2], shape[3]
        g_cpu = torch.Generator()
        g_cpu.manual_seed(15)
        torch.manual_seed(15)
        log_probs = (
            torch.randn(T, N, C, dtype=dtype).log_softmax(2).detach().requires_grad_()
        )
        targets = torch.randint(1, N, (N, S), dtype=torch.long, generator=g_cpu)
        input_lengths = torch.full((N,), T, dtype=torch.long)
        target_lengths = torch.randint(1, S, (N,), dtype=torch.long, generator=g_cpu)
        _test_loss_ctc(log_probs, targets, input_lengths, target_lengths, dtype)
        g_cpu = torch.Generator()
        g_cpu.manual_seed(15)
        torch.manual_seed(15)
