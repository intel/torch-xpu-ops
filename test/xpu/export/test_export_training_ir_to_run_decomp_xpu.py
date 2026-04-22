# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from PyTorch
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

# Owner(s): ["module: intel"]

import sys
from pathlib import Path

torch_xpu_ops_path = Path(__file__).resolve().parents[1]
if str(torch_xpu_ops_path) not in sys.path:
    sys.path.insert(0, str(torch_xpu_ops_path))

from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(patch_test_case=False):
    __pytorch_test_dir = Path(__file__).resolve().parents[5] / "test"
    if str(__pytorch_test_dir) not in sys.path:
        sys.path.insert(0, str(__pytorch_test_dir))

    import torch

    from export import test_export, testing
    from torch.testing._internal.common_utils import run_tests

import unittest

from torch.testing._internal.triton_utils import requires_xpu_and_triton


# Override CUDA-only tests on TestExport with XPU variants before the dynamic
# TrainingIRToRunDecomp* classes are generated from it. Each override mirrors
# the source semantics but targets device="xpu".


@requires_xpu_and_triton
@testing.expectedFailureCppRuntime
def _test_export_associative_scan_symbol_dim(self):
    from torch._higher_order_ops import associative_scan as _assoc_scan

    device = torch.device("xpu")
    combine_mode = "pointwise"

    dim1 = torch.export.Dim("dim0", min=5, max=15)
    xs = torch.ones(3, 10, 2, device=device)

    class Foo(torch.nn.Module):
        def combine_fn(self, x, y):
            return x + y

        def forward(self, x):
            return _assoc_scan(self.combine_fn, x, 2, combine_mode=combine_mode)

    ep = torch.export.export(Foo(), (xs,), dynamic_shapes={"x": {1: dim1}})
    module_out = Foo()(xs)
    self.assertTrue(torch.allclose(ep.module()(xs), module_out))


@requires_xpu_and_triton
@testing.expectedFailureCppRuntime
def _test_export_associative_scan_symbol_scandim(self):
    from torch._higher_order_ops import associative_scan as _assoc_scan

    device = torch.device("xpu")
    combine_mode = "pointwise"

    dim1 = torch.export.Dim("dim0", min=5, max=15)
    xs = torch.ones(3, 10, 2, device=device)

    class Foo(torch.nn.Module):
        def combine_fn(self, x, y):
            return x + y

        def forward(self, x):
            return _assoc_scan(self.combine_fn, x, 1, combine_mode=combine_mode)

    ep = torch.export.export(Foo(), (xs,), dynamic_shapes={"x": {1: dim1}})
    module_out = Foo()(xs)
    self.assertTrue(torch.allclose(ep.module()(xs), module_out))


@requires_xpu_and_triton
def _test_export_associative_scan_lifted_buffers(self):
    from torch._higher_order_ops import associative_scan as _assoc_scan

    if "cpp_runtime_nonstrict" in self.id():
        self.skipTest("TODO Unexpected success in OSS but not in fbcode.")

    device = torch.device("xpu")
    combine_mode = "pointwise"

    class A(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.buffer = torch.nn.Buffer(torch.ones(3, 2, device=device))

        def forward(self):
            return self.buffer.cos()

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = A()

        def combine_fn(self, x, y):
            return (x + y) * self.a()

        def forward(self, x):
            return _assoc_scan(self.combine_fn, x, 1, combine_mode=combine_mode)

    inp = torch.ones(3, 10, 2, device=device)
    ep = torch.export.export(M(), (inp,))
    epm = ep.module()

    self.assertTrue(torch.allclose(epm(inp), M()(inp)))

    for gm in epm.named_modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        self.assertEqual(
            len([node for node in gm.graph.nodes if node.op == "placeholder"]), 1
        )


@unittest.skipIf(not torch.xpu.is_available(), "Test requires XPU.")
def _test_exception(self):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)
            self.register_buffer("buffer", torch.ones(4, 4))
            self.register_buffer("param", torch.ones(4, 4))

        def forward(self, x):
            token_ids = torch.randint(0, 10, (4,), device=x.device)
            embedded = self.embedding(token_ids).sum()
            return self.buffer.sum() + self.param.sum() + x.sum() + embedded

    class BarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = Model()

        def forward(self, x):
            if "xpu" in str(x.device):
                mod = self.mod.to(x.device)
                return mod(x)
            else:
                return x.sum()

    class BarBar(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mod = BarModel()

        def forward(self, x):
            with torch.amp.autocast(device_type="xpu"):
                y = self.mod(x)
            return y

    with torch.no_grad():
        with self.assertRaisesRegex(RuntimeError, "Couldn't swap Embedding.weight"):
            _ = torch.export.export(
                BarBar(),
                (),
                {"x": torch.randn(4, 4, 4, device="xpu")},
                strict=False,
            ).module()


test_export.TestExport.test_export_associative_scan_symbol_dim = (
    _test_export_associative_scan_symbol_dim
)
test_export.TestExport.test_export_associative_scan_symbol_scandim = (
    _test_export_associative_scan_symbol_scandim
)
test_export.TestExport.test_export_associative_scan_lifted_buffers = (
    _test_export_associative_scan_lifted_buffers
)
test_export.TestExport.test_exception = _test_exception

test_classes = {}


def mocked_training_ir_to_run_decomp_export_strict(*args, **kwargs):
    if "strict" in kwargs:
        ep = torch.export.export(*args, **kwargs)
    else:
        ep = torch.export.export(*args, **kwargs, strict=True)
    return ep.run_decompositions({})


def mocked_training_ir_to_run_decomp_export_non_strict(*args, **kwargs):
    ep = torch.export.export(*args, **kwargs)

    return ep.run_decompositions({})


def make_dynamic_cls(cls, strict):
    if strict:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "TrainingIRToRunDecompExport",
            test_export.TRAINING_IR_DECOMP_STRICT_SUFFIX,
            mocked_training_ir_to_run_decomp_export_strict,
            xfail_prop="_expected_failure_training_ir_to_run_decomp",
        )
    else:
        test_class = testing.make_test_cls_with_mocked_export(
            cls,
            "TrainingIRToRunDecompExportNonStrict",
            test_export.TRAINING_IR_DECOMP_NON_STRICT_SUFFIX,
            mocked_training_ir_to_run_decomp_export_non_strict,
            xfail_prop="_expected_failure_training_ir_to_run_decomp_non_strict",
        )

    test_classes[test_class.__name__] = test_class
    globals()[test_class.__name__] = test_class
    test_class.__module__ = __name__
    return test_class


with XPUPatchForImport(patch_test_case=False):
    tests = [
        test_export.TestDynamismExpression,
        test_export.TestExport,
    ]
    for test in tests:
        make_dynamic_cls(test, True)
        make_dynamic_cls(test, False)
    del test


if __name__ == "__main__":
    run_tests()