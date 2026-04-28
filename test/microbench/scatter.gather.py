import torch
from torch.profiler import profile, ProfilerActivity

device = "xpu"
backward = True

# Define shapes for scatter/gather testing
# (input_shape, index_shape, dim_to_scatter_gather)
shape_list = [
    ((4096, 8192), (4096, 8192), 1),  # Simple 2D case
    ((2, 4096, 320), (2, 4096, 1), 2), # Scatter/Gather along the last dim
    ((512, 3136, 128), (512, 1, 128), 1), # Scatter/Gather along the middle dim
    ((128, 49, 196, 1024), (128, 49, 196, 1), 3), # 4D case, scatter/gather last dim
]

for shape_config in shape_list:
    input_shape, index_shape, dim_to_operate = shape_config

    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        # Generate input tensor
        input_tensor = torch.randn(input_shape, device=device, dtype=dtype)

        # Generate index tensor for gather/scatter
        # Ensure indices are within valid bounds for the dimension
        max_idx_val = input_tensor.shape[dim_to_operate]
        index_tensor = torch.randint(0, max_idx_val, index_shape, device=device, dtype=torch.int64)

        # Generate source tensor for scatter
        # Its shape should match index_tensor in the dimension being scattered into,
        # and input_tensor in other dimensions.
        scatter_source_shape = list(input_tensor.shape)
        for i, dim_size in enumerate(index_shape):
            if i == dim_to_operate:
                scatter_source_shape[i] = dim_size
        scatter_source = torch.randn(scatter_source_shape, device=device, dtype=dtype)

        if backward:
            input_tensor.requires_grad_(True)
            scatter_source.requires_grad_(True)

        # Warm-up phase
        # Gather operation
        gathered_output_warmup = torch.gather(input_tensor, dim_to_operate, index_tensor)
        if backward:
            gy_gather = torch.empty_like(gathered_output_warmup)
            gathered_output_warmup.backward(gy_gather)

        # Scatter operation (using out-of-place scatter_ to ensure a fresh tensor for profiling)
        scattered_output_warmup = input_tensor.clone().scatter_(dim_to_operate, index_tensor, scatter_source)
        if backward:
            gy_scatter = torch.empty_like(scattered_output_warmup)
            scattered_output_warmup.backward(gy_scatter)

        print(
            "---"
        )
        print(
            "Testing Scatter/Gather -- input shape:",
            input_shape,
            "; index shape:",
            index_shape,
            "; datatype:",
            dtype,
            "; dim:",
            dim_to_operate,
            "; backward:",
            backward,
        )
        print(
            "---"
        )

        # Profiling phase
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.XPU], record_shapes=True
        ) as prof:
            for i in range(20):
                # Gather operation
                gathered_output = torch.gather(input_tensor, dim_to_operate, index_tensor)
                if backward:
                    gy_gather = torch.empty_like(gathered_output)
                    gathered_output.backward(gy_gather)

                # Scatter operation
                # We clone input_tensor each time to avoid modifying the same tensor
                # across iterations, which could affect profiling if in-place ops are used.
                scattered_output = input_tensor.clone().scatter_(dim_to_operate, index_tensor, scatter_source)
                if backward:
                    gy_scatter = torch.empty_like(scattered_output)
                    scattered_output.backward(gy_scatter)

        print(prof.key_averages().table(sort_by="xpu_time_total"))