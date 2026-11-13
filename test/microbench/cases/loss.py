# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Loss functions."""

from ..case import (
    F,
    FLOAT_DTYPES,
    full,
    OpBench,
    randint,
    randn,
    State,
    table_bench,
    torch,
)


class Loss(OpBench):
    """``F.<op>(x, target, reduction=r)`` fwd/bwd; default inputs: two randn."""

    family = "loss"
    passes = ("fwd", "bwd")

    def setup(self, p, device):
        x, t = (
            randn(p["shape"], p["dtype"], device),
            randn(p["shape"], p["dtype"], device),
        )
        return State(x=x, t=t, r=p["reduction"])

    def fwd(self, s):
        return getattr(F, self.op)(s.x, s.t, reduction=s.r)


class MseLoss(Loss):
    op = "mse_loss"
    expected = {"fwd": ("mse_loss",), "bwd": ("mse_loss_backward",)}
    params = [
        dict(shape=(8192, 8192), dtype=dt, reduction=r)
        for dt in FLOAT_DTYPES
        for r in ("none", "mean")
    ]
    pr = [dict(shape=(8192, 8192), dtype="bf16", reduction="mean")]


class SmoothL1Loss(Loss):
    op = "smooth_l1_loss"
    expected = {"fwd": ("smooth_l1_loss",), "bwd": ("smooth_l1_loss_backward",)}
    params = [
        dict(shape=s, dtype=dt, reduction=r)
        for s in [(8732, 8732), (8192, 8732)]
        for dt in FLOAT_DTYPES
        for r in ("none", "mean")
    ]
    pr = [dict(shape=(8192, 8732), dtype="bf16", reduction="mean")]


class L1Loss(Loss):
    """No kernel of its own: ``abs(input - target)`` plus the reduction;
    backward is ``sgn(diff) * grad``."""

    op = "l1_loss"
    composite = True
    params = [
        dict(shape=s, dtype=dt, reduction=r)
        for r in ("none", "mean")
        for s in [(8732, 8732), (8192, 8732)]
        for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(8192, 8732), dtype="bf16", reduction="mean")]

    def expected_ops(self, p, pass_):
        if pass_ == "bwd":
            return ("sgn", "mul")
        return ("sub", "abs") + (("mean",) if p["reduction"] == "mean" else ())


class BinaryCrossEntropy(Loss):
    op = "binary_cross_entropy"
    expected = {
        "fwd": ("binary_cross_entropy",),
        "bwd": ("binary_cross_entropy_backward",),
    }
    params = [
        dict(shape=s, dtype=dt, reduction=r)
        for s in [(8733, 8733), (8733, 513), (513, 8733), (8192, 8192)]
        for dt in FLOAT_DTYPES
        for r in ("none", "mean", "sum")
    ]
    pr = [dict(shape=(8192, 8192), dtype="bf16", reduction="mean")]

    def setup(self, p, device):
        x = torch.sigmoid(randn(p["shape"], p["dtype"], device))
        t = randint(0, 2, p["shape"], device, dtype="int64").to(x.dtype)
        return State(x=x, t=t, r=p["reduction"])


class MultilabelMarginLoss(Loss):
    """Targets: ``labels`` distinct classes per row, then -1."""

    op = "multilabel_margin_loss"
    expected = {
        "fwd": ("multilabel_margin_loss_forward",),
        "bwd": ("multilabel_margin_loss_backward",),
    }
    params = [
        dict(shape=(8192, 8192), dtype=dt, reduction=r, labels=5)
        for dt in FLOAT_DTYPES
        for r in ("none", "mean")
    ]
    pr = [dict(shape=(8192, 8192), dtype="bf16", reduction="mean", labels=5)]

    def setup(self, p, device):
        (n, c), k = p["shape"], p["labels"]
        t = torch.full((n, c), -1, dtype=torch.long)
        t[:, :k] = torch.rand(n, c).argsort(dim=1)[:, :k]
        return State(
            x=randn(p["shape"], p["dtype"], device), t=t.to(device), r=p["reduction"]
        )


class NllLoss(Loss):
    op = "nll_loss"
    expected = {"fwd": ("nll_loss_forward",), "bwd": ("nll_loss_backward",)}
    params = [
        dict(shape=(8192, 8192), dtype=dt, reduction="mean") for dt in FLOAT_DTYPES
    ]
    pr = [dict(shape=(8192, 8192), dtype="bf16", reduction="mean")]

    def setup(self, p, device):
        t = torch.arange(p["shape"][0], device=device)
        return State(x=randn(p["shape"], p["dtype"], device), t=t, r=p["reduction"])


class CtcLoss(OpBench):
    """fp32 only. Lengths are host lists, as with the IntArrayRef overload: device
    length tensors would add a device-to-host copy and sync to the timed region.
    Target lengths are capped at T // 2 so every alignment is feasible."""

    op = "ctc_loss"
    family = "loss"
    passes = ("fwd", "bwd")
    expected = {"fwd": ("_ctc_loss",), "bwd": ("_ctc_loss_backward",)}
    # (T, N, C, S): input length, batch, classes, max target length
    params = [
        dict(shape=s, dtype="fp32")
        for s in [(32, 32, 32, 16), (128, 128, 128, 128), (8, 8, 4, 8)]
    ]
    pr = [dict(shape=(128, 128, 128, 128), dtype="fp32")]

    def setup(self, p, device):
        t, n, c, s_len = p["shape"]
        log_probs = torch.randn(t, n, c).log_softmax(2).to(device)
        targets = torch.randint(1, c, (n, s_len)).to(device)
        target_lengths = torch.randint(1, max(1, min(s_len, t // 2)) + 1, (n,)).tolist()
        return State(x=log_probs, tg=targets, il=[t] * n, tl=target_lengths)

    def fwd(self, s):
        return F.ctc_loss(s.x, s.tg, s.il, s.tl)

    def expect_inputs(self, p, pass_):
        t, n, c, _ = p["shape"]
        return [dict(shape=(t, n, c), dtype="fp32")] if pass_ == "fwd" else None


def nll2d_inputs(p, device):
    """Target uniform in [0, C) with 5% ``ignore_index``."""
    x = randn(p["shape"], p["dtype"], device)
    target = randint(0, p["shape"][1], p["target_shape"], device)
    target[(torch.rand(p["target_shape"]) < 0.05).to(device)] = p["ignore_index"]
    return State(
        x=x, target=target, reduction=p["reduction"], ignore_index=p["ignore_index"]
    )


NLL2D = dict(
    shape=(64, 20005, 1, 128), target_shape=(64, 1, 128), reduction=1, ignore_index=0
)
NllLoss2dForward = table_bench(
    "nll_loss2d",
    [NLL2D],
    pr=[0],
    family="loss",
    expected={"fwd": ("nll_loss2d_forward",)},
    setup=lambda self, p, device: nll2d_inputs(p, device),
    fwd=lambda self, s: torch.ops.aten.nll_loss2d_forward(
        s.x, s.target, None, s.reduction, s.ignore_index
    ),
)


def nll2d_backward_inputs(self, p, device):
    """Scalar grad_output and total_weight (mean reduction)."""
    s = nll2d_inputs(p, device)
    s.grad, s.total_weight = (
        randn((), p["dtype"], device),
        full((), 1000.0, p["dtype"], device),
    )
    return s


NllLoss2dBackward = table_bench(
    "nll_loss2d",
    [
        NLL2D,
        dict(
            shape=(4, 2, 640, 959),
            target_shape=(4, 640, 959),
            reduction=1,
            ignore_index=-100,
        ),
    ],
    pr=[0],
    family="loss",
    passes=("bwd",),
    setup=nll2d_backward_inputs,
    bwd=lambda self, s: torch.ops.aten.nll_loss2d_backward(
        s.grad, s.x, s.target, None, s.reduction, s.ignore_index, s.total_weight
    ),
)
