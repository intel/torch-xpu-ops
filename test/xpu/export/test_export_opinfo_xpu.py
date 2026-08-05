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

import os
import subprocess
import sys

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import run_tests, TestCase

for op in op_db:
    if (
        op.name == "nn.functional.batch_norm"
        and op.variant_test_name == "without_cudnn"
    ):
        # Assign XPU-specific dtypes
        if hasattr(op, "_dispatch_dtypes"):
            cuda_dtypes = op._dispatch_dtypes.get("cuda")
            if cuda_dtypes:
                op._dispatch_dtypes["xpu"] = cuda_dtypes
        # Keep others like 'disablecuDNN' which are usually harmless or handled
        if hasattr(op, "supported_device_types"):
            op.supported_device_types = op.supported_device_types.union({"xpu"})
        op.decorators = tuple(
            d for d in op.decorators if "onlyCUDA" not in getattr(d, "__name__", str(d))
        )

        break

selected_ops = {
    "__getitem__",
    "nn.functional.batch_norm",
    "nn.functional.conv2d",
    "nn.functional.instance_norm",
    "nn.functional.multi_margin_loss",
    "nn.functional.scaled_dot_product_attention",
    "nonzero",
}
selected_op_db = [op for op in op_db if op.name in selected_ops]


class TestExportOnFakeCuda(TestCase):
    # In CI, this test runs on a XPU machine with XPU build
    # We set ZE_AFFINITY_MASK="" to simulate a CPU machine with XPU build
    # Running this on all ops in op_db is too slow, so we only run on a selected subset
    @ops(selected_op_db, allowed_dtypes=(torch.float,))
    def test_fake_export(self, device, dtype, op):
        test_script = f"""\
import torch
import itertools
from torch.testing._internal.common_methods_invocations import op_db
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils import _pytree as pytree

ops = [op for op in op_db if op.name == "{op.name}"]
assert len(ops) > 0

for op in ops:
    sample_inputs_itr = op.sample_inputs("cpu", torch.float, requires_grad=False)

    mode = FakeTensorMode(allow_non_fake_inputs=True)

    target_device = "xpu:0"

    def to_fake_device(x):
        return x.to(target_device)

    # Limit to first 100 inputs so tests don't take too long
    for sample_input in itertools.islice(sample_inputs_itr, 100):
        args = tuple([sample_input.input] + list(sample_input.args))
        kwargs = sample_input.kwargs

        # hack to skip non-tensor in args, as export doesn't support it
        if any(not isinstance(arg, torch.Tensor) for arg in args):
            continue

        if "device" in kwargs:
            kwargs["device"] = target_device

        with mode:
            args, kwargs = pytree.tree_map_only(
                torch.Tensor, to_fake_device, (args, kwargs)
            )

            class Module(torch.nn.Module):
                def forward(self, *args):
                    return op.op(*args, **kwargs)

            m = Module()

            ep = torch.export.export(m, args)

            for node in ep.graph.nodes:
                if node.op == "call_function":
                    fake_tensor = node.meta.get("val", None)
                    if isinstance(fake_tensor, FakeTensor):
                        assert fake_tensor.device == torch.device(target_device)
"""
        r = (
            (
                subprocess.check_output(
                    [sys.executable, "-c", test_script],
                    env={**os.environ, "ZE_AFFINITY_MASK": ""},
                )
            )
            .decode("ascii")
            .strip()
        )
        self.assertEqual(r, "")


instantiate_device_type_tests(
    TestExportOnFakeCuda, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
