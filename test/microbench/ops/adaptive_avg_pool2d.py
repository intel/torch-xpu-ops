# ops/adaptive_avg_pool.py
import torch
from core.runner import normalize_dtype


def run_op(config, device):
    """config keys: shape[N,C,H,W], out[OH,OW], dtype, channels_last(bool), backward(bool)"""
    N, C, H, W = config["shape"]
    output_size = tuple(config["output_size"])
    dtype = normalize_dtype(config.get("datatype", torch.float32))
    channels_last = config.get("channels_last", False)
    backward = config.get("backward", True)

    x = torch.randn(N, C, H, W, requires_grad=True).to(device=device, dtype=dtype)
    if channels_last:
        x = x.to(memory_format=torch.channels_last)

    output = torch.nn.AdaptiveAvgPool2d(output_size)(x)

    if backward:
        Wout = output_size[0]
        Hout = output_size[1]
        grad = torch.rand([C, Hout, Wout], requires_grad=True).to(
            device=device, dtype=dtype
        )
        output[0].backward(grad)

def get_default_cases():
    base_shapes = [
        ([8, 512, 32, 32], [7, 7]),
        ([8, 256, 56, 56], [14, 14]),
    ]
    dtypes = [torch.bfloat16, torch.float16, torch.float32]
    cases = []
    for shape, out in base_shapes:
        for dtype in dtypes:
            for channels_last in [False, True]:
                cases.append({
                    "shape": shape,
                    "datatype": dtype,
                    "channels_last": channels_last,
                    "output_size": out,
                    "backward": True,
                })
    return cases
