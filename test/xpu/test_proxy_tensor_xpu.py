# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts
from torch.testing._internal.common_utils import run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception:
    from .xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_proxy_tensor import (
        TestGenericProxyTensorFake,
        TestGenericProxyTensorReal,
        TestGenericProxyTensorSymbolic,
        TestSymbolicTracing,
    )


# --- TestGenericProxyTensor CUDA-only overrides ---

def _amp_cache_xpu(self):
    layer = torch.nn.Conv2d(3, 3, 3).xpu()

    def f(x, w):
        return torch.nn.functional.conv2d(x, w, stride=layer.stride)

    inp = torch.randn(4, 3, 10, 10, device="xpu")
    with torch.autocast("xpu"):
        out_graph = make_fx(f)(inp, layer.weight).graph
        out_graph2 = make_fx(f)(inp, layer.weight).graph

    self.assertEqual(len(out_graph.nodes), len(out_graph2.nodes))
    for a, b in zip(out_graph.nodes, out_graph2.nodes):
        self.assertEqual(a.op, b.op)


def _T244632748_xpu(self):
    class TestModule(torch.nn.Module):
        def forward(self, x):
            return x + (x.shape[0] * 2)

    mod = TestModule()
    sample = torch.randn((5, 5)).to("xpu")
    dim0 = torch.export.Dim.DYNAMIC(max=100)
    dynamic_shapes = {"x": (dim0, torch.export.Dim.STATIC)}
    ep = torch.export.export(mod, (sample,), dynamic_shapes=dynamic_shapes)
    gm = ep.module()
    symint = list(gm.graph.nodes)[3].meta["val"]
    list(gm.graph.nodes)[3].replace_all_uses_with(symint)
    gm.graph.eliminate_dead_code()

    torch._inductor.aot_compile(
        gm, (sample,), options={"fx_wrapper": True, "compile_threads": 1}
    )


for _cls in (
    TestGenericProxyTensorReal,
    TestGenericProxyTensorFake,
    TestGenericProxyTensorSymbolic,
):
    _cls.test_amp_cache = _amp_cache_xpu
    _cls.test_T244632748 = _T244632748_xpu
del _cls


# --- TestSymbolicTracing CUDA-only overrides ---

def _cpu_scalar_cuda_xpu(self):
    # Extracted from wave2vec2. Despite the test name, the asserted graph
    # has no device info; switch the input device to xpu.
    def f(a, b):
        return (a * b) @ b

    r = str(
        make_fx(f, tracing_mode="symbolic")(
            torch.tensor(1.0), torch.randn(2, 2, device="xpu")
        ).code
    ).strip()
    self.assertExpectedInline(
        r,
        """\
def forward(self, a_1, b_1):
    mul = torch.ops.aten.mul.Tensor(a_1, b_1);  a_1 = None
    mm = torch.ops.aten.mm.default(mul, b_1);  mul = b_1 = None
    return mm""",
    )


def _view_divisibility_unbacked_relatively_prime_xpu(self):
    def f(x):
        i0 = x.item()
        torch._check(i0 > 0)
        torch._check(i0 <= 448)
        return torch.zeros(256 * i0).view(-1, 447)

    make_fx(f, tracing_mode="symbolic")(torch.tensor(256 * 447, device="xpu"))


def _unbacked_unify_dependency_violation_xpu(self):
    def f(x1, x2, x3, y):
        z1 = x1.item()
        torch._check(z1 // 9 == 1)
        z2 = x2.item()
        z3 = x3.item()
        torch._check(z1 == z2 + z3)
        return y * 2

    gm = make_fx(f, tracing_mode="symbolic")(
        torch.tensor(10, device="xpu"),
        torch.tensor(5, device="xpu"),
        torch.tensor(5, device="xpu"),
        torch.randn(1, device="xpu"),
    )
    insert_deferred_runtime_asserts(gm, gm.shape_env, "test")
    gm.recompile()
    self.assertEqual(
        gm(
            torch.tensor(12, device="xpu"),
            torch.tensor(6, device="xpu"),
            torch.tensor(6, device="xpu"),
            torch.tensor([1.0], device="xpu"),
        ),
        torch.tensor([2.0], device="xpu"),
    )
    with self.assertRaises(RuntimeError):
        gm(
            torch.tensor(20, device="xpu"),
            torch.tensor(10, device="xpu"),
            torch.tensor(10, device="xpu"),
            torch.tensor([1.0], device="xpu"),
        )


# test_unbacked_unify_guard_transitivity is marked @unittest.expectedFailure
# in the source; preserve that semantics on XPU.
import unittest  # noqa: E402


@unittest.expectedFailure
def _unbacked_unify_guard_transitivity_xpu(self):
    def f(x1, x2, y):
        z1 = torch.zeros(x1.item())
        z2 = torch.zeros(x2.item())
        torch._check(z1.size(0) == z2.size(0))
        torch._check(z2.size(0) == y.size(0))
        if z1.size(0) == 4:
            return y * 2
        else:
            return y + 2

    gm = make_fx(f, tracing_mode="symbolic")(
        torch.tensor(10, device="xpu"),
        torch.tensor(10, device="xpu"),
        torch.randn(10, device="xpu"),
    )
    insert_deferred_runtime_asserts(gm, gm.shape_env, "test")
    gm.recompile()
    str(gm.code).strip()


TestSymbolicTracing.test_cpu_scalar_cuda = _cpu_scalar_cuda_xpu
TestSymbolicTracing.test_view_divisibility_unbacked_relatively_prime = (
    _view_divisibility_unbacked_relatively_prime_xpu
)
TestSymbolicTracing.test_unbacked_unify_guard_transitivity = (
    _unbacked_unify_guard_transitivity_xpu
)
TestSymbolicTracing.test_unbacked_unify_dependency_violation = (
    _unbacked_unify_dependency_violation_xpu
)


if __name__ == "__main__":
    run_tests()
