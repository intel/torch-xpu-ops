# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
import torch._inductor.decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_out_dtype_op import TestOutDtypeOp


def _out_dtype_inductor_decomp_trace_xpu(self) -> None:
    def func(x, w):
        return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

    w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="xpu")
    x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="xpu")

    # Check that make_fx with inductor decomps produces _int_mm
    decomp_table = torch._inductor.decomposition.select_decomp_table()
    gm = make_fx(func, decomp_table, tracing_mode="symbolic")(x, w)
    self.assertExpectedInline(
        gm.code.strip(),
        """\
def forward(self, x_1, w_1):
    _int_mm = torch.ops.aten._int_mm.default(x_1, w_1);  x_1 = w_1 = None
    return _int_mm""",
    )


def _out_dtype_int_mm_default_trace_xpu(self) -> None:
    def func(x, w):
        return out_dtype(torch.ops.aten.mm.default, torch.int32, x, w)

    w = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="xpu")
    x = torch.randint(-128, 127, (32, 32), dtype=torch.int8, device="xpu")

    # By default, out_dtype is preserved in the trace
    gm = make_fx(func, tracing_mode="symbolic")(x, w)
    self.assertExpectedInline(
        gm.code.strip(),
        """\
def forward(self, x_1, w_1):
    out_dtype = torch.ops.higher_order.out_dtype(torch.ops.aten.mm.default, torch.int32, x_1, w_1);  x_1 = w_1 = None
    return out_dtype""",
    )


# Override the CUDA-only tests with XPU variants
TestOutDtypeOp.test_out_dtype_inductor_decomp_trace = (
    _out_dtype_inductor_decomp_trace_xpu
)
TestOutDtypeOp.test_out_dtype_int_mm_default_trace = (
    _out_dtype_int_mm_default_trace_xpu
)


if __name__ == "__main__":
    run_tests()
