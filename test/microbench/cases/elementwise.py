# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Elementwise ops (Loops.h path) with real-model shapes and strides."""

from ..case import FLOAT_DTYPES, OpBench, randn, State, table_bench, torch

# no nan/inf from out-of-domain values
POSITIVE = {"x": ("uniform", 0.5, 2.0)}
# 1% nan, 1% +inf, 1% -inf
NONFINITE = {"x": ("nonfinite", 0.01)}


def unary(op, cases, pr=(), **kw):
    fwd = lambda self, s: getattr(torch, op)(s.x)  # noqa: E731
    return table_bench(op, cases, pr, family="elementwise", fwd=fwd, **kw)


def binary(op, cases, pr=(), **kw):
    fwd = lambda self, s: getattr(torch, op)(s.x, s.other)  # noqa: E731
    return table_bench(op, cases, pr, family="elementwise", fwd=fwd, **kw)


Abs = unary(
    "abs",
    [
        dict(shape=(2048, 2048), dtype="int64"),
        dict(shape=(32, 1024, 1024)),
        dict(shape=(10, 2, 352, 351), in_stride=(247104, 1, 702, 2)),
        dict(shape=(1000, 81)),
    ],
    pr=[0],
)
Cos = unary("cos", [dict(shape=(1024, 512)), dict(shape=(1, 512, 32))])
Sin = unary("sin", [dict(shape=(1024, 512)), dict(shape=(1, 512, 32))])
Exp = unary(
    "exp",
    [
        dict(shape=(4902912, 1)),
        dict(shape=(64, 12, 64, 64, 128)),
        dict(shape=(128, 3, 12, 16, 2), in_stride=(48960, 16320, 1360, 85, 1)),
    ],
    pr=[0],
)
Floor = unary("floor", [dict(shape=(16000,))])
Neg = unary(
    "neg",
    [
        dict(shape=(16, 512, 768)),
        dict(shape=(64, 12, 64, 64, 128)),
        dict(shape=(64, 384, 14, 14), in_stride=(75264, 1, 5376, 384)),
        dict(shape=()),
    ],
    pr=[0],
)
Sgn = unary(
    "sgn",
    [
        dict(shape=(10, 3, 352, 352)),
        dict(shape=(10, 2, 351, 352), in_stride=(247104, 1, 704, 2)),
    ],
    pr=[0],
)
Log = unary(
    "log", [dict(shape=(2048, 2048)), dict(shape=(128, 128))], pr=[0], values=POSITIVE
)
Log2 = unary("log2", [dict(shape=(16000,))], values=POSITIVE)
Sqrt = unary("sqrt", [dict(shape=(25536, 80)), dict(shape=())], pr=[0], values=POSITIVE)
Rsqrt = unary(
    "rsqrt",
    [
        dict(shape=(64, 12, 64, 64, 1)),
        dict(shape=(128, 12, 64, 64, 1)),
        dict(shape=(4, 1024, 1)),
    ],
    pr=[0],
    values=POSITIVE,
)
Reciprocal = unary(
    "reciprocal",
    [dict(shape=(512,), dtype="fp32"), dict(shape=(16, 16, 2), dtype="fp64")],
    values=POSITIVE,
)
BitwiseNot = unary(
    "bitwise_not",
    [dict(shape=(16, 1, 512, 512), dtype="bool"), dict(shape=(), dtype="bool")],
    pr=[0],
)
Isinf = unary(
    "isinf",
    [
        dict(shape=(32, 1024, 512)),
        dict(shape=(32, 512, 1024)),
        dict(shape=(16, 1, 512)),
    ],
    pr=[0],
    values=NONFINITE,
)
NanToNum = unary(
    "nan_to_num",
    [dict(shape=(256, 1, 158)), dict(shape=(256, 1, 1543))],
    values=NONFINITE,
)

Eq = binary(
    "eq",
    [
        dict(shape=(16, 16, 128, 128), other_shape=(), other_dtype="fp32"),
        dict(shape=(32, 1024, 1024), other_shape=(32, 1024, 1024)),
        dict(shape=(1000, 81), other_shape=(1000, 81)),
        dict(shape=(16, 2), other=0, dtype="int64"),
    ],
    pr=[0],
)
Ne = binary(
    "ne",
    [
        dict(shape=(32, 1024, 1024), other=float("inf")),
        dict(
            shape=(128, 12, 64, 64, 1), other_shape=(128, 12, 64, 1, 128), dtype="int64"
        ),
        dict(shape=(1000, 81), other=float("inf")),
    ],
    pr=[0],
)
Lt = binary(
    "lt",
    [
        dict(shape=(2048, 2048), other=8, dtype="int64"),
        dict(shape=(16, 16, 128, 128), other_shape=(), other_dtype="fp32"),
        dict(shape=(), other=0.0),
    ],
    pr=[0],
)
Le = binary(
    "le",
    [
        dict(shape=(1, 1, 1, 1024), other_shape=(1, 1, 1024, 1), dtype="int64"),
        dict(shape=(1, 1, 1, 128), other_shape=(1, 1, 128, 1), dtype="int64"),
    ],
    pr=[0],
)
Gt = binary(
    "gt",
    [
        dict(shape=(2048, 2048), other=0, dtype="int64"),
        dict(shape=(25536, 80), other=0.2),
    ],
    pr=[0],
)
Ge = binary(
    "ge",
    [dict(shape=(90, 246, 640), other=0.5), dict(shape=(81, 480, 640), other=0.5)],
    pr=[0],
)
Maximum = binary(
    "maximum",
    [dict(shape=(16, 16, 128, 128), other_shape=(), other_dtype="fp32")],
    pr=[0],
)
Minimum = binary(
    "minimum",
    [
        dict(shape=(1024, 1024), other_shape=(1024, 1024), dtype="int64"),
        dict(shape=(8, 1, 1, 448), other_shape=(8, 12, 64, 448), other_dtype="fp32"),
        dict(shape=(128, 128), other_shape=(128, 128), dtype="int64"),
    ],
    pr=[0],
)
Mul = binary(
    "mul",
    [
        dict(
            shape=(16, 12, 512, 64),
            in_stride=(393216, 64, 768, 1),
            other=0.3535533905932738,
        ),
        dict(shape=(1000, 81), other_shape=(1000, 81), dtype="bool"),
    ],
    pr=[0],
)
Div = binary(
    "div",
    [
        dict(shape=(3, 800, 1067), other_shape=(3, 1, 1)),
        dict(shape=(128, 16, 512, 512), other=8.0),
        dict(shape=(128, 2048, 7, 7), in_stride=(2048, 1, 0, 0), other=49),
        dict(shape=(), other=2),
    ],
    pr=[0],
)
Sub = binary(
    "sub",
    [
        dict(shape=(3, 800, 1067), other_shape=(3, 1, 1)),
        dict(shape=(64, 12, 64, 64, 128), other_shape=(64, 12, 64, 64, 1)),
        dict(
            shape=(4902912,), in_stride=(4,), other_shape=(4902912,), other_stride=(4,)
        ),
    ],
    pr=[0],
)
Rsub = binary(
    "rsub",
    [
        dict(shape=(8, 12, 64, 448), other=1.0),
        dict(
            shape=(8, 12, 60, 64, 192),
            in_stride=(9142272, 761856, 12288, 192, 1),
            other=1.0,
        ),
        dict(shape=(4, 1, 1, 1024), other=1.0),
    ],
    pr=[0],
)
BitwiseAnd = binary(
    "bitwise_and",
    [
        dict(shape=(1000,), other_shape=(1000,), dtype="bool"),
        dict(shape=(1024,), other_shape=(1024,), other_dtype="bool", dtype="int64"),
    ],
)
BitwiseOr = binary(
    "bitwise_or", [dict(shape=(1024,), other_shape=(1024,), dtype="bool")]
)

Addcmul = table_bench(
    "addcmul",
    [
        dict(
            shape=(1, 1, 384), tensor1_shape=(1, 1, 384), tensor2_shape=(4096, 196, 384)
        ),
        dict(
            shape=(1, 1, 384), tensor1_shape=(1, 1, 384), tensor2_shape=(256, 196, 384)
        ),
    ],
    pr=[0],
    family="elementwise",
    fwd=lambda self, s: torch.addcmul(s.x, s.tensor1, s.tensor2),
)
Clamp = table_bench(
    "clamp",
    [
        dict(shape=(4902912, 1), max=4.135166556742356),
        dict(
            shape=(32, 512, 1024),
            min_shape=(),
            min_dtype="fp32",
            max_shape=(),
            max_dtype="fp32",
        ),
        dict(shape=(80000,), in_stride=(4,), min=0, max=800),
    ],
    pr=[0],
    family="elementwise",
    defaults={"min": None},
    fwd=lambda self, s: torch.clamp(s.x, s.min, s.max),
)
# condition 50% True
Where = table_bench(
    "where",
    [
        dict(
            shape=(1024, 1024),
            input_shape=(1024, 1024),
            input_dtype="int64",
            other_shape=(1024, 1024),
            other_dtype="int64",
            dtype="bool",
        ),
        dict(
            shape=(64, 12, 64, 64, 128),
            in_dtype="bool",
            input_shape=(64, 12, 64, 64, 128),
            other_shape=(),
        ),
        dict(
            shape=(4, 1, 1024, 1024),
            in_stride=(0, 1048576, 1024, 1),
            in_dtype="bool",
            input_shape=(),
            other_shape=(),
        ),
        dict(shape=(), in_dtype="bool", input_shape=(), other_shape=()),
    ],
    pr=[0],
    family="elementwise",
    fwd=lambda self, s: torch.where(s.x, s.input, s.other),
)
# token ids uniform in [0, 50257)
Isin = table_bench(
    "isin",
    [
        dict(shape=(50257,), test_elements_shape=(1,), dtype="int64"),
        dict(shape=(1024, 32), test_elements_shape=(), dtype="int64"),
    ],
    pr=[0],
    family="elementwise",
    values={"x": ("int", 0, 50257), "test_elements": ("int", 0, 50257)},
    fwd=lambda self, s: torch.isin(s.x, s.test_elements),
)
# Tensor ** Scalar: bases uniform in [0.5, 2); Scalar ** Tensor: exponents in [0, 1)
Pow = table_bench(
    "pow",
    [
        dict(shape=(4, 1024, 512), exponent=2),
        dict(shape=(64, 512, 16384), exponent=3.0),
        dict(shape=(4, 1024, 1), exponent=3),
        dict(base=10000, shape=(512,)),
    ],
    pr=[0],
    family="elementwise",
    values={
        "x": lambda p: ("uniform", 0.0, 1.0) if "base" in p else ("uniform", 0.5, 2.0)
    },
    defaults={"base": None, "exponent": None},
    fwd=lambda self, s: torch.pow(s.x, s.exponent)
    if s.base is None
    else torch.pow(s.base, s.x),
)


class Add(OpBench):
    op = "add"
    family = "elementwise"
    expected = {"fwd": ("add",)}
    CASES = [
        dict(shape=(8192, 8192), other="tensor"),  # contiguous, vectorized
        dict(shape=(8192, 8192), other="tensor", in_stride=(20000, 2)),  # strided
        dict(shape=(8190, 8190), other="tensor"),  # not a multiple of the vector width
        dict(shape=(8192, 8192), other=0.5),  # scalar
    ]
    params = [dict(c, dtype=dt) for c in CASES for dt in FLOAT_DTYPES]
    pr = [dict(CASES[0], dtype="bf16"), dict(CASES[1], dtype="bf16")]

    def setup(self, p, device):
        stride = p.get("in_stride")
        a = randn(p["shape"], p["dtype"], device, stride=stride)
        b = (
            randn(p["shape"], p["dtype"], device, stride=stride)
            if p["other"] == "tensor"
            else p["other"]
        )
        return State(a=a, b=b)

    def fwd(self, s):
        return torch.add(s.a, s.b)


class Remainder(OpBench):
    """Forward only: the gradient w.r.t. the input is the incoming gradient
    itself (no kernel). Shapes with 1.07G elements are halved on dim 0."""

    op = "remainder"
    family = "elementwise"
    expected = {"fwd": ("remainder",)}
    params = [
        dict(shape=s, dtype=dt, divisor=d)
        for s in [(512, 1024, 1024), (6, 7, 3, 2), (4097, 8193, 4, 4)]
        for dt in FLOAT_DTYPES
        for d in (2, -1.5, 3)
    ]
    pr = [dict(shape=(512, 1024, 1024), dtype="bf16", divisor=-1.5)]

    def setup(self, p, device):
        return State(x=randn(p["shape"], p["dtype"], device), d=p["divisor"])

    def fwd(self, s):
        return torch.remainder(s.x, s.d)
