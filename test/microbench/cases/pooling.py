# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Pooling, each case contiguous and channels_last."""

from ..case import F, FLOAT_DTYPES, nd_input, OpBench, State, torch


def variants(cases):
    return [
        dict(c, dtype=dt, channels_last=cl)
        for c in cases
        for dt in FLOAT_DTYPES
        for cl in (False, True)
    ]


class Pool(OpBench):
    """``s.x`` from shape/dtype/channels_last; other params become ``s.<key>``."""

    family = "pooling"
    passes = ("fwd", "bwd")

    def setup(self, p, device):
        x = nd_input(p["shape"], p["dtype"], device, p["channels_last"])
        return State(
            x=x,
            **{
                k: v
                for k, v in p.items()
                if k not in ("shape", "dtype", "channels_last")
            },
        )


POOL2D = [
    dict(shape=(16, 24, 112, 112), kernel_size=3, stride=2),
    dict(shape=(16, 1984, 7, 7), kernel_size=(3, 2), stride=(2, 1)),
    dict(shape=(64, 1024, 112, 112), kernel_size=6, stride=4),
    dict(shape=(4, 2048, 224, 224), kernel_size=3, stride=2),  # batch 16 -> 4 (memory)
]


class AvgPool2d(Pool):
    op = "avg_pool2d"
    expected = {"fwd": ("avg_pool2d",), "bwd": ("avg_pool2d_backward",)}
    params = variants(POOL2D)
    pr = [dict(POOL2D[0], dtype="bf16", channels_last=True)]

    def fwd(self, s):
        return F.avg_pool2d(s.x, s.kernel_size, stride=s.stride)


class MaxPool2d(AvgPool2d):
    op = "max_pool2d"
    expected = {
        "fwd": ("max_pool2d_with_indices",),
        "bwd": ("max_pool2d_with_indices_backward",),
    }

    def fwd(self, s):
        return F.max_pool2d(s.x, s.kernel_size, stride=s.stride, return_indices=True)


class AvgPool3d(Pool):
    op = "avg_pool3d"
    expected = {"fwd": ("avg_pool3d",), "bwd": ("avg_pool3d_backward",)}
    params = variants(
        [
            dict(shape=(16, 24, 28, 19, 19), kernel_size=3, stride=2),
            dict(shape=(16, 1984, 7, 7, 7), kernel_size=(3, 2, 2), stride=(2, 1, 2)),
            dict(shape=(64, 1024, 14, 14, 14), kernel_size=6, stride=4),
        ]
    )
    pr = [
        dict(
            shape=(16, 24, 28, 19, 19),
            kernel_size=3,
            stride=2,
            dtype="bf16",
            channels_last=False,
        )
    ]

    def fwd(self, s):
        return F.avg_pool3d(s.x, s.kernel_size, stride=s.stride)


class MaxPool3d(Pool):
    op = "max_pool3d"
    expected = {
        "fwd": ("max_pool3d_with_indices",),
        "bwd": ("max_pool3d_with_indices_backward",),
    }
    params = variants(
        [
            dict(shape=s, kernel_size=2)
            for s in [
                (16, 32, 64, 64, 64),
                (1, 4, 144, 144, 144),
                (512, 512, 12, 12, 12),
            ]
        ]
    )
    pr = [
        dict(
            shape=(16, 32, 64, 64, 64), kernel_size=2, dtype="bf16", channels_last=False
        )
    ]

    def fwd(self, s):
        return F.max_pool3d(s.x, s.kernel_size, return_indices=True)


class AdaptiveAvgPool2d(Pool):
    op = "adaptive_avg_pool2d"
    expected = {
        "fwd": ("_adaptive_avg_pool2d",),
        "bwd": ("_adaptive_avg_pool2d_backward",),
    }
    params = variants(
        [
            dict(shape=(8, 512, 32, 32), output_size=(7, 7)),
            dict(shape=(8, 256, 56, 56), output_size=(14, 14)),
        ]
    )
    pr = [
        dict(
            shape=(8, 256, 56, 56),
            output_size=(14, 14),
            dtype="bf16",
            channels_last=True,
        )
    ]

    def fwd(self, s):
        return F.adaptive_avg_pool2d(s.x, s.output_size)


class AdaptiveMaxPool2d(Pool):
    op = "adaptive_max_pool2d"
    expected = {
        "fwd": ("adaptive_max_pool2d",),
        "bwd": ("adaptive_max_pool2d_backward",),
    }
    params = variants(
        [
            dict(shape=(8, 512, 7, 7), output_size=(1, 1)),
            dict(shape=(8, 512, 32, 32), output_size=(7, 7)),
            dict(shape=(8, 256, 56, 56), output_size=(14, 14)),
        ]
    )
    pr = [
        dict(
            shape=(8, 256, 56, 56),
            output_size=(14, 14),
            dtype="bf16",
            channels_last=False,
        )
    ]

    def fwd(self, s):
        return F.adaptive_max_pool2d(s.x, s.output_size, return_indices=True)


class FractionalMaxPool2d(Pool):
    """The pooling regions (``_random_samples``) are drawn in setup, not by an
    RNG kernel in the timed region."""

    op = "fractional_max_pool2d"
    expected = {
        "fwd": ("fractional_max_pool2d",),
        "bwd": ("fractional_max_pool2d_backward",),
    }
    params = variants(
        [
            # batch 32 -> 16 (memory)
            dict(shape=(16, 32, 1024, 1024), output_size=(512, 512)),
            dict(shape=(32, 4, 128, 128), output_size=(64, 64)),
            dict(shape=(1, 3, 1200, 1200), output_size=(600, 600)),
            dict(shape=(512, 512, 28, 28), output_size=(14, 14)),
        ]
    )
    pr = [
        dict(
            shape=(32, 4, 128, 128),
            output_size=(64, 64),
            dtype="bf16",
            channels_last=False,
        )
    ]

    def setup(self, p, device):
        s = super().setup(p, device)
        n, c = p["shape"][:2]
        s.samples = torch.rand(n, c, len(p["shape"]) - 2, dtype=s.x.dtype).to(device)
        return s

    def fwd(self, s):
        return getattr(F, self.op)(
            s.x,
            2,
            output_size=s.output_size,
            return_indices=True,
            _random_samples=s.samples,
        )


class FractionalMaxPool3d(FractionalMaxPool2d):
    op = "fractional_max_pool3d"
    expected = {
        "fwd": ("fractional_max_pool3d",),
        "bwd": ("fractional_max_pool3d_backward",),
    }
    params = variants(
        [
            # batch 32 -> 8 (memory)
            dict(shape=(8, 32, 128, 128, 128), output_size=(64, 64, 64)),
            dict(shape=(1, 3, 144, 144, 144), output_size=(72, 72, 72)),
            dict(shape=(512, 512, 12, 12, 12), output_size=(6, 6, 6)),
        ]
    )
    pr = [
        dict(
            shape=(1, 3, 144, 144, 144),
            output_size=(72, 72, 72),
            dtype="bf16",
            channels_last=False,
        )
    ]


class MaxUnpool2d(Pool):
    """The max pooling that produces the indices runs in setup. Backward has no
    dedicated kernel: autograd gathers the gradient."""

    op = "max_unpool2d"
    expected = {"fwd": ("max_unpool2d",), "bwd": ("gather",)}
    params = variants(
        [
            dict(shape=s, kernel_size=2)
            for s in [(4, 64, 128, 128), (4, 65, 128, 128), (8, 128, 128, 128)]
        ]
    )
    pr = [
        dict(shape=(8, 128, 128, 128), kernel_size=2, dtype="bf16", channels_last=False)
    ]

    def setup(self, p, device):
        x = nd_input(p["shape"], p["dtype"], device, p["channels_last"])
        pool = getattr(F, self.op.replace("unpool", "pool"))
        y, idx = pool(x, p["kernel_size"], return_indices=True)
        return State(
            x=y.detach(), idx=idx, k=p["kernel_size"], size=tuple(p["shape"][2:])
        )

    def fwd(self, s):
        return getattr(F, self.op)(s.x, s.idx, s.k, output_size=s.size)

    def expect_inputs(self, p, pass_):
        if pass_ != "fwd":
            return None
        n, c, *spatial = p["shape"]
        return [dict(shape=(n, c, *(d // 2 for d in spatial)), dtype=p["dtype"])]


class MaxUnpool3d(MaxUnpool2d):
    op = "max_unpool3d"
    expected = {"fwd": ("max_unpool3d",), "bwd": ("gather",)}
    params = variants(
        [
            dict(shape=s, kernel_size=2)
            for s in [(2, 32, 64, 64, 64), (4, 33, 64, 64, 64), (16, 32, 32, 32, 32)]
        ]
    )
    pr = [
        dict(
            shape=(2, 32, 64, 64, 64), kernel_size=2, dtype="bf16", channels_last=False
        )
    ]
