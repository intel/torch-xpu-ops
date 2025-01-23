import torch

from torch.profiler import profile, record_function, ProfilerActivity

def maxUnpool2d(shape, dtype, channels_last, backward):
    N, C, H, W = int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])
    kernel_size = 2

    pool = torch.nn.MaxPool2d(kernel_size, return_indices=True)
    unpool = torch.nn.MaxUnpool2d(kernel_size)
    #torch.manual_seed(20)

    if channels_last:
        input = torch.randn([N, C, H, W]).to(memory_format=torch.channels_last).to(device="cpu", dtype=torch.float32)
    else:
        input = torch.randn([N, C, H, W]).to(device="cpu", dtype=torch.float32)
    output, indices = pool(input)

    #pool.to(device="xpu", dtype=dtype)
    #unpool.to(device="xpu", dtype=dtype)
    if channels_last:
        x_dpcpp = output.to(memory_format=torch.channels_last).to(device="xpu", dtype=dtype)
        indices_dpcpp = indices.to(memory_format=torch.channels_last).to(device="xpu", dtype=torch.int64)
    else:
        x_dpcpp = output.to(device="xpu", dtype=dtype)
        indices_dpcpp = indices.to(device="xpu", dtype=torch.int64)

    if backward:
        x_dpcpp.requires_grad_(True)
        if channels_last:
            grad_dpcpp = torch.randn([N, C, H, W]).to(memory_format=torch.channels_last).to(device="xpu", dtype=dtype)
        else:
            grad_dpcpp = torch.randn([N, C, H, W]).to(device="xpu", dtype=dtype)

    y_dpcpp = unpool(x_dpcpp, indices_dpcpp, output_size=torch.Size([N,C,H,W])).to("xpu")

    if backward:
        y_dpcpp.backward(grad_dpcpp)

if __name__ == "__main__":
    dtype = torch.bfloat16
    dtype = torch.float32
    backward = True
    #for channels_last in [False, True]:
    #    for shape in [[4,64,128,128],[4,65,128,128],[8,128,128,128]]:
    for channels_last in [False]:
        for shape in [[4,64,128,128]]:
            print("======================================")
            print("channels_last is %s, backward is %s, shape is %s" % (str(channels_last), str(backward),str(shape)))

            # warm up
            maxUnpool2d(shape, dtype, channels_last, backward=backward)
            maxUnpool2d(shape, dtype, channels_last, backward=backward)
            maxUnpool2d(shape, dtype, channels_last, backward=backward)

            # go
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.XPU],) as prof:
                for i in range(1):
                    maxUnpool2d(shape, dtype, channels_last, backward=backward)
            print(prof.key_averages().table(sort_by="xpu_time_total"))
