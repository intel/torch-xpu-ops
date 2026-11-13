# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Activations. Backward cases call the aten ``*_backward`` op directly with
``s.x`` as grad_output."""

from ..case import F, table_bench, torch


def fwd(op, cases, call, pr=(), **kw):
    return table_bench(
        op, cases, pr, family="activation", fwd=lambda self, s: call(s), **kw
    )


def bwd(op, cases, call, pr=(), **kw):
    return table_bench(
        op,
        cases,
        pr,
        family="activation",
        passes=("bwd",),
        bwd=lambda self, s: call(s),
        **kw,
    )


def nhwc(n, c, h, w):
    """Shape and channels_last strides."""
    return dict(shape=(n, c, h, w), in_stride=(c * h * w, 1, w * c, c))


def grad_and_input(case):
    """Backward params: grad_output and input with the same shape/strides."""
    extra = {"input_shape": case["shape"]}
    if "in_stride" in case:
        extra["input_stride"] = case["in_stride"]
    return dict(case, **extra)


Gelu = fwd(
    "gelu",
    [dict(shape=(16, 512, 3072)), nhwc(128, 768, 12, 12)],
    lambda s: F.gelu(s.x),
    pr=[0],
)
GeluBackward = bwd(
    "gelu",
    [
        grad_and_input(dict(shape=(16, 512, 3072))),
        grad_and_input(nhwc(128, 768, 12, 12)),
        grad_and_input(dict(shape=(256, 1, 128))),
    ],
    lambda s: torch.ops.aten.gelu_backward(s.x, s.input),
    pr=[0],
)

Hardsigmoid = fwd(
    "hardsigmoid",
    [
        nhwc(2048, 120, 1, 1),
        nhwc(2048, 960, 1, 1),
    ],
    lambda s: F.hardsigmoid(s.x),
)
HardsigmoidBackward = bwd(
    "hardsigmoid",
    [
        dict(
            shape=(256, 120, 1, 1),
            input_shape=(256, 120, 1, 1),
            input_stride=(120, 1, 120, 120),
        ),
        dict(
            shape=(512, 960, 1, 1),
            input_shape=(512, 960, 1, 1),
            input_stride=(960, 1, 960, 960),
        ),
    ],
    lambda s: torch.ops.aten.hardsigmoid_backward(s.x, s.input),
)

Hardswish = fwd(
    "hardswish",
    [nhwc(2048, 16, 112, 112), nhwc(2048, 32, 1, 1)],
    lambda s: F.hardswish(s.x, inplace=True),
    pr=[0],
    inplace=True,
)
HardswishBackward = bwd(
    "hardswish",
    [
        grad_and_input(nhwc(256, 16, 112, 112)),
        grad_and_input(nhwc(1024, 32, 56, 56)),
        grad_and_input(nhwc(256, 48, 1, 1)),
    ],
    lambda s: torch.ops.aten.hardswish_backward(s.x, s.input),
    pr=[0],
)

RELU6 = dict(min_val=0.0, max_val=6.0)
# in place; clamping again is a no-op, so no reset
Hardtanh = fwd(
    "hardtanh",
    [dict(nhwc(1024, 384, 14, 14), **RELU6), dict(nhwc(1024, 144, 56, 56), **RELU6)],
    lambda s: F.hardtanh(s.x, s.min_val, s.max_val, inplace=True),
    pr=[0],
    idempotent=True,
)
HardtanhBackward = bwd(
    "hardtanh",
    [
        dict(grad_and_input(nhwc(256, 32, 112, 112)), **RELU6),
        dict(grad_and_input(nhwc(256, 144, 56, 56)), **RELU6),
    ],
    lambda s: torch.ops.aten.hardtanh_backward(s.x, s.input, s.min_val, s.max_val),
    pr=[0],
)

LeakyRelu = fwd(
    "leaky_relu",
    [nhwc(512, 128, 32, 32), nhwc(512, 64, 128, 128)],
    lambda s: F.leaky_relu(s.x, inplace=True),
    pr=[0],
    inplace=True,
)
LeakyReluBackward = bwd(
    "leaky_relu",
    [
        dict(
            shape=(128, 256, 16, 16),
            in_stride=(131072, 1, 8192, 512),
            input_shape=(128, 256, 16, 16),
            input_stride=(65536, 1, 4096, 256),
            negative_slope=0.01,
            self_is_result=True,
        ),
        dict(
            grad_and_input(nhwc(128, 64, 128, 128)),
            negative_slope=0.01,
            self_is_result=True,
        ),
    ],
    lambda s: torch.ops.aten.leaky_relu_backward(
        s.x, s.input, s.negative_slope, s.self_is_result
    ),
    pr=[0],
)

# relu_ is idempotent, so no reset
Relu = fwd(
    "relu",
    [
        dict(nhwc(128, 64, 112, 112), inplace=True),
        nhwc(512, 768, 32, 32),
        dict(nhwc(1024, 128, 1, 1), inplace=True),
    ],
    lambda s: torch.relu_(s.x) if s.inplace else torch.relu(s.x),
    pr=[0],
    idempotent=True,
)

Sigmoid = fwd(
    "sigmoid",
    [
        dict(shape=(25536, 80), inplace=True),
        dict(shape=(2048, 50, 3072)),
        dict(shape=(10, 1, 352, 352), in_stride=(619520, 1, 1760, 5)),
    ],
    lambda s: torch.sigmoid_(s.x) if s.inplace else torch.sigmoid(s.x),
    pr=[0],
)
SigmoidBackward = bwd(
    "sigmoid",
    [
        dict(shape=(256, 6, 2048), output_shape=(256, 6, 2048)),
        dict(shape=(256, 50, 3072), output_shape=(256, 50, 3072)),
        dict(
            shape=(10, 1, 352, 352),
            output_shape=(10, 1, 352, 352),
            output_stride=(123904, 1, 352, 1),
        ),
    ],
    lambda s: torch.ops.aten.sigmoid_backward(s.x, s.output),
    pr=[0],
)

Silu = fwd(
    "silu",
    [
        dict(nhwc(512, 480, 14, 14), inplace=True),
        dict(shape=(2048, 196, 768), in_stride=(301056, 1536, 1)),
        dict(nhwc(512, 28, 1, 1), inplace=True),
    ],
    lambda s: F.silu(s.x, inplace=s.inplace),
    pr=[0],
)
SiluBackward = bwd(
    "silu",
    [
        grad_and_input(nhwc(128, 480, 14, 14)),
        grad_and_input(nhwc(128, 256, 64, 64)),
        grad_and_input(nhwc(128, 28, 1, 1)),
    ],
    lambda s: torch.ops.aten.silu_backward(s.x, s.input),
    pr=[0],
)
