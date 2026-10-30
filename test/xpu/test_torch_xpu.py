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

# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfXPU,
    instantiate_device_type_tests,
    onlyNativeDeviceTypes,
    onlyXPU,
)
from torch.testing._internal.common_dtype import (
    all_types_and,
    all_types_and_complex_and,
    floating_types_and,
    get_all_math_dtypes,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    TestCase,
)

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx

with XPUImportCtx(False):
    import test_torch
    from test_torch import (
        TestDevicePrecision,
        TestTensorDeviceOps,
        TestTorch,
        TestTorchDeviceType,
        TestViewOps,
    )


# ======================================================================
# Extra XPU dtypes
# ======================================================================

TestTorchDeviceType.test_bernoulli_p = dtypesIfXPU(*floating_types_and(torch.half))(
    TestTorchDeviceType.test_bernoulli_p
)

TestTorchDeviceType.test_bernoulli_self = dtypesIfXPU(
    *all_types_and(torch.bool, torch.half)
)(TestTorchDeviceType.test_bernoulli_self)

TestTorchDeviceType.test_diff = dtypesIfXPU(
    *all_types_and_complex_and(torch.half, torch.bool)
)(TestTorchDeviceType.test_diff)

TestTorchDeviceType.test_addcdiv = dtypesIfXPU(
    *set(get_all_math_dtypes("xpu")) | {torch.half}
)(TestTorchDeviceType.test_addcdiv)

TestTorchDeviceType.test_multinomial_cpu = dtypesIfXPU(
    torch.float, torch.double, torch.half, torch.bfloat16
)(TestTorchDeviceType.test_multinomial_cpu)

TestDevicePrecision.test_from_sequence = dtypesIfXPU(
    torch.half,
    torch.float,
    torch.double,
    torch.int8,
    torch.short,
    torch.int,
    torch.long,
    torch.uint8,
)(TestDevicePrecision.test_from_sequence)


# ======================================================================
# scatter_reduce: https://github.com/intel/torch-xpu-ops/issues/5376
# ======================================================================


def _scatter_allow_reduce(self, device, dtype, reduceop):
    device_type = torch.device(device).type
    return device_type not in ("cuda", "xpu") or (
        reduceop == "multiply" and dtype.is_floating_point
    )


TestTorchDeviceType.scatter_allow_reduce = _scatter_allow_reduce

for _name in (
    "test_scatter_reduce_operations_to_large_input",
    "test_scatter_reduce_scalar",
    "test_scatter_reduce_non_unique_index",
):
    _test = getattr(TestTorchDeviceType, _name)
    # Drop the upstream decorateIf(expectedFailure) for XPU complex dtypes.
    vars(_test).pop("parametrize_fn", None)
    dtypesIfXPU(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))(
        _test
    )


# ======================================================================
# Tests whose upstream body does not support XPU yet
# ======================================================================


# https://github.com/intel/torch-xpu-ops/issues/5286
@onlyNativeDeviceTypes
@dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16))
def _test_masked_scatter(self, device, dtype):
    dt = dtype
    num_copy, num_dest = 3, 10
    dest = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=dt, device=device)
    dest2 = dest.clone()
    dest_ones = dest.clone()
    dest_ones_expected = dest.clone()
    src = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dt, device=device)
    src_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dt, device=device)
    mask = torch.tensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0), dtype=torch.bool, device=device)

    dest.masked_scatter_(mask, src)
    j = 0
    for i in range(num_dest):
        if mask[i]:
            dest2[i] = src[j]
            dest_ones_expected[i] = src_ones[j]
            j += 1
    self.assertEqual(dest, dest2, atol=0, rtol=0)

    dest_ones.masked_scatter_(mask, src_ones)
    self.assertEqual(dest_ones, dest_ones_expected, atol=0, rtol=0)

    # Bound checking in CUDA and XPU is done inside a kernel
    # in order to avoid synchronization, but this means
    # we can not clear the failures. So there is no way
    # to test it then recover.
    if self.device_type not in ("cuda", "xpu"):
        # make src smaller. this should fail
        src = torch.zeros(num_copy - 1, dtype=dt, device=device)
        with self.assertRaises(RuntimeError):
            dest.masked_scatter_(mask, src)

    # empty tensor
    dest = torch.empty((5, 0, 5), dtype=dt, device=device)
    mask = torch.ones_like(dest, dtype=torch.bool, device=device)
    src = torch.empty((0,), dtype=dt, device=device)
    dest.masked_scatter_(mask, src)

    dest = torch.empty((5, 0, 5), dtype=dt, device=device)
    mask = torch.ones((5, 1, 5), dtype=torch.bool, device=device)
    src = torch.empty((0,), dtype=dt, device=device)
    dest.masked_scatter_(mask, src)


TestTorchDeviceType.test_masked_scatter = _test_masked_scatter


# https://github.com/intel/torch-xpu-ops/issues/5304
@dtypes(torch.float32, torch.int32)
def _test_nondeterministic_alert_histc(self, device, dtype):
    a = torch.tensor([], device=device, dtype=dtype)
    for op_call in [torch.histc, torch.Tensor.histc]:
        self.check_nondeterministic_alert(
            lambda: op_call(a, min=0, max=3),
            "_histc_xpu with floating point input",
            dtype.is_floating_point,
        )


TestTorchDeviceType.test_nondeterministic_alert_histc = (
    _test_nondeterministic_alert_histc
)


@onlyXPU
def _test_memory_format_cpu_and_cuda_ops(self, device):
    def get_generator(memory_format, shape):
        def input_generator_fn(device):
            return torch.randn(shape, device=device, dtype=torch.float32).contiguous(
                memory_format=memory_format
            )

        return input_generator_fn

    def transformation_cpu_fn(tensor, **kwargs):
        return tensor.cpu(**kwargs)

    def transformation_xpu_fn(tensor, **kwargs):
        return tensor.xpu(**kwargs)

    formats_shapes = (
        (torch.channels_last, (4, 3, 8, 8)),
        (torch.channels_last_3d, (4, 3, 8, 8, 8)),
    )

    for mf, shape in formats_shapes:
        self._test_memory_format_transformations(
            "xpu",
            get_generator(mf, shape),
            transformation_cpu_fn,
            mf,
            default_is_preserve=True,
        )
        self._test_memory_format_transformations(
            "cpu",
            get_generator(mf, shape),
            transformation_xpu_fn,
            mf,
            default_is_preserve=True,
        )


TestTorchDeviceType.test_memory_format_cpu_and_cuda_ops = (
    _test_memory_format_cpu_and_cuda_ops
)

TestTorchDeviceType.test_addcmul_cuda_errors_with_cpu_scalars = (
    test_torch.TestTorchCUDA.test_addcmul_cuda_errors_with_cpu_scalars
)


# https://github.com/intel/torch-xpu-ops/issues/5661 (empty inputs crash on XPU)
def _test_split_with_sizes_copy_out(self, device):
    shape = (30, 40, 50)
    x = torch.rand(*shape, device=device)
    cases = [
        (0, [3, 7, 8, 12]),
        (1, [3, 7, 10, 20]),
        (-2, [3, 7, 10, 20]),
        (2, [3, 7, 10, 12, 18]),
        (-1, [3, 7, 10, 12, 18]),
        (2, [3, 7, 10, 0, 30]),
    ]
    for dim, split_sizes in cases:
        views = x.split_with_sizes(split_sizes, dim=dim)
        expects = [v.clone() for v in views]
        out = [torch.zeros_like(v) for v in views]
        for expect, t in zip(expects, out):
            if expect.numel() != 0:
                self.assertFalse(expect.eq(t).all().item())

        torch.split_with_sizes_copy(x, split_sizes, dim=dim, out=out)
        for expect, t in zip(expects, out):
            self.assertTrue(expect.eq(t).all().item())


TestTorchDeviceType.test_split_with_sizes_copy_out = _test_split_with_sizes_copy_out

# https://github.com/intel/torch-xpu-ops/issues/5309 (passes when run from test/xpu)
TestTorch.test_terminate_handler_on_crash = (
    TestTorch.test_terminate_handler_on_crash.__wrapped__
)


# ======================================================================
# Instantiate test classes for XPU execution
# ======================================================================

instantiate_device_type_tests(TestViewOps, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(
    TestTensorDeviceOps, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestTorchDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestDevicePrecision, globals(), only_for="xpu", allow_xpu=True
)
instantiate_parametrized_tests(TestTorch)


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
