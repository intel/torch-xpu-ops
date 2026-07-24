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

import unittest
import warnings

import numpy as np
import torch
from torch.testing._internal.common_device_type import (
    deviceCountAtLeast,
    dtypes,
    dtypesIfXPU,
    expectedFailureMeta,
    instantiate_device_type_tests,
    largeTensorTest,
    onlyNativeDeviceTypes,
    onlyOn,
)
from torch.testing._internal.common_dtype import (
    all_types_and,
    all_types_and_complex_and,
    get_all_dtypes,
)
from torch.testing._internal.common_utils import (
    do_test_empty_full,
    IS_FBCODE,
    IS_PPC,
    IS_SANDCASTLE,
    run_tests,
    suppress_warnings,
    TestCase,
)

try:
    from xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx
except Exception:
    from .xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx

with XPUImportCtx(False):
    from test_tensor_creation_ops import (
        TestAsArray,
        TestBufferProtocol,
        TestFromBlob,
        TestLikeTensorCreation,
        TestRandomTensorCreation,
        TestTensorCreation,
    )


# ======================================================================
# Override upstream test methods for XPU migration adjustments
# ======================================================================


@suppress_warnings
@onlyNativeDeviceTypes
@deviceCountAtLeast(1)
def _test_tensor_device(self, devices):
    device_type = torch.device(devices[0]).type
    if device_type == "cpu":
        self.assertEqual("cpu", torch.tensor(5).device.type)
        self.assertEqual(
            "cpu", torch.ones((2, 3), dtype=torch.float32, device="cpu").device.type
        )
        self.assertEqual(
            "cpu", torch.ones((2, 3), dtype=torch.float32, device="cpu:0").device.type
        )
        self.assertEqual(
            "cpu",
            torch.tensor(
                torch.ones((2, 3), dtype=torch.float32), device="cpu:0"
            ).device.type,
        )
        self.assertEqual(
            "cpu", torch.tensor(np.random.randn(2, 3), device="cpu").device.type
        )
    if device_type == "cuda":
        self.assertEqual("cuda:0", str(torch.tensor(5).cuda(0).device))
        self.assertEqual("cuda:0", str(torch.tensor(5).cuda("cuda:0").device))
        self.assertEqual(
            "cuda:0", str(torch.tensor(5, dtype=torch.int64, device=0).device)
        )
        self.assertEqual(
            "cuda:0", str(torch.tensor(5, dtype=torch.int64, device="cuda:0").device)
        )
        self.assertEqual(
            "cuda:0",
            str(
                torch.tensor(
                    torch.ones((2, 3), dtype=torch.float32), device="cuda:0"
                ).device
            ),
        )

        self.assertEqual(
            "cuda:0", str(torch.tensor(np.random.randn(2, 3), device="cuda:0").device)
        )

        for device in devices:
            with torch.cuda.device(device):
                device_string = "cuda:" + str(torch.cuda.current_device())
                self.assertEqual(
                    device_string,
                    str(torch.tensor(5, dtype=torch.int64, device="cuda").device),
                )

        with self.assertRaises(RuntimeError):
            torch.tensor(5).cuda("cpu")
        with self.assertRaises(RuntimeError):
            torch.tensor(5).cuda("cpu:0")

        if len(devices) > 1:
            self.assertEqual("cuda:1", str(torch.tensor(5).cuda(1).device))
            self.assertEqual("cuda:1", str(torch.tensor(5).cuda("cuda:1").device))
            self.assertEqual(
                "cuda:1", str(torch.tensor(5, dtype=torch.int64, device=1).device)
            )
            self.assertEqual(
                "cuda:1",
                str(torch.tensor(5, dtype=torch.int64, device="cuda:1").device),
            )
            self.assertEqual(
                "cuda:1",
                str(
                    torch.tensor(
                        torch.ones((2, 3), dtype=torch.float32), device="cuda:1"
                    ).device
                ),
            )

            self.assertEqual(
                "cuda:1",
                str(torch.tensor(np.random.randn(2, 3), device="cuda:1").device),
            )

    if device_type == "xpu":
        self.assertEqual("xpu:0", str(torch.tensor(5).xpu(0).device))
        self.assertEqual("xpu:0", str(torch.tensor(5).xpu("xpu:0").device))
        self.assertEqual(
            "xpu:0", str(torch.tensor(5, dtype=torch.int64, device=0).device)
        )
        self.assertEqual(
            "xpu:0", str(torch.tensor(5, dtype=torch.int64, device="xpu:0").device)
        )
        self.assertEqual(
            "xpu:0",
            str(
                torch.tensor(
                    torch.ones((2, 3), dtype=torch.float32), device="xpu:0"
                ).device
            ),
        )

        self.assertEqual(
            "xpu:0", str(torch.tensor(np.random.randn(2, 3), device="xpu:0").device)
        )

        for device in devices:
            with torch.xpu.device(device):
                device_string = "xpu:" + str(torch.xpu.current_device())
                self.assertEqual(
                    device_string,
                    str(torch.tensor(5, dtype=torch.int64, device="xpu").device),
                )

        with self.assertRaises(RuntimeError):
            torch.tensor(5).xpu("cpu")
        with self.assertRaises(RuntimeError):
            torch.tensor(5).xpu("cpu:0")

        if len(devices) > 1:
            self.assertEqual("xpu:1", str(torch.tensor(5).xpu(1).device))
            self.assertEqual("xpu:1", str(torch.tensor(5).xpu("xpu:1").device))
            self.assertEqual(
                "xpu:1", str(torch.tensor(5, dtype=torch.int64, device=1).device)
            )
            self.assertEqual(
                "xpu:1", str(torch.tensor(5, dtype=torch.int64, device="xpu:1").device)
            )
            self.assertEqual(
                "xpu:1",
                str(
                    torch.tensor(
                        torch.ones((2, 3), dtype=torch.float32), device="xpu:1"
                    ).device
                ),
            )

            self.assertEqual(
                "xpu:1", str(torch.tensor(np.random.randn(2, 3), device="xpu:1").device)
            )


TestTensorCreation.test_tensor_device = _test_tensor_device


@onlyNativeDeviceTypes
@unittest.skipIf(
    IS_PPC,
    "Test is broken on PowerPC, see https://github.com/pytorch/pytorch/issues/39671",
)
@dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
def _test_float_to_int_conversion_finite(self, device, dtype):
    min = torch.finfo(torch.float).min
    max = torch.finfo(torch.float).max

    # Note: CUDA max float -> integer conversion is divergent on some dtypes
    vals = (min, -2, -1.5, -0.5, 0, 0.5, 1.5, 2, max)
    refs = None
    if self.device_type in ("cuda", "xpu"):
        if torch.version.hip or torch.version.xpu:
            # HIP min float -> int64 conversion is divergent
            # XPU min float -> int8 conversion is divergent
            # XPU min float -> int16 conversion is divergent
            vals = (-2, -1.5, -0.5, 0, 0.5, 1.5, 2)
        else:
            vals = (min, -2, -1.5, -0.5, 0, 0.5, 1.5, 2)
    elif dtype == torch.uint8:
        # Note: CPU max float -> uint8 conversion is divergent
        vals = (min, -2, -1.5, -0.5, 0, 0.5, 1.5, 2)
        # Note: numpy -2.0 or -1.5 -> uint8 conversion is undefined
        #       see https://github.com/pytorch/pytorch/issues/97794
        refs = (0, 254, 255, 0, 0, 0, 1, 2)

    self._float_to_int_conversion_helper(vals, device, dtype, refs)


TestTensorCreation.test_float_to_int_conversion_finite = (
    _test_float_to_int_conversion_finite
)


@onlyNativeDeviceTypes
def _test_empty_full(self, device):
    torch_device = torch.device(device)
    device_type = torch_device.type

    dtypes = get_all_dtypes(
        include_half=False, include_bfloat16=False, include_complex32=True
    )
    if device_type == "cpu":
        do_test_empty_full(self, dtypes, torch.strided, torch_device)
    if device_type in ("cuda", "xpu"):
        do_test_empty_full(self, dtypes, torch.strided, None)
        do_test_empty_full(self, dtypes, torch.strided, torch_device)


TestTensorCreation.test_empty_full = _test_empty_full


# Removed onlyCPU + added XPU if branches
def _test_as_tensor(self, device):
    device_type = torch.device(device).type

    # from python data
    x = [[0, 1], [2, 3]]
    self.assertEqual(torch.tensor(x), torch.as_tensor(x))
    self.assertEqual(
        torch.tensor(x, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32)
    )

    # python data with heterogeneous types
    z = [0, "torch"]
    with self.assertRaisesRegex(TypeError, "invalid data type"):
        torch.tensor(z)
        torch.as_tensor(z)

    # python data with self-referential lists
    z = [0]
    z += [z]
    with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
        torch.tensor(z)
        torch.as_tensor(z)

    z = [[1, 2], z]
    with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
        torch.tensor(z)
        torch.as_tensor(z)

    # from tensor (doesn't copy unless type is different)
    y = torch.tensor(x)
    self.assertIs(y, torch.as_tensor(y))
    self.assertIsNot(y, torch.as_tensor(y, dtype=torch.float32))

    if device_type in ("cuda", "xpu"):
        self.assertIsNot(y, torch.as_tensor(y, device=device_type))
        y_device = y.to(device_type)
        self.assertIs(y_device, torch.as_tensor(y_device))
        self.assertIs(y_device, torch.as_tensor(y_device, device=device_type))

    # doesn't copy
    for dtype in [np.float64, np.int64, np.int8, np.uint8]:
        n = np.random.rand(5, 6).astype(dtype)
        n_astensor = torch.as_tensor(n)
        self.assertEqual(torch.tensor(n), n_astensor)
        n_astensor[0][0] = 25.7
        self.assertEqual(torch.tensor(n), n_astensor)

    # changing dtype causes copy
    n = np.random.rand(5, 6).astype(np.float32)
    n_astensor = torch.as_tensor(n, dtype=torch.float64)
    self.assertEqual(torch.tensor(n, dtype=torch.float64), n_astensor)
    n_astensor[0][1] = 250.8
    self.assertNotEqual(torch.tensor(n, dtype=torch.float64), n_astensor)

    # changing device causes copy
    if device_type in ("cuda", "xpu"):
        n = np.random.rand(5, 6)
        n_astensor = torch.as_tensor(n, device=device_type)
        self.assertEqual(torch.tensor(n, device=device_type), n_astensor)
        n_astensor[0][2] = 250.9
        self.assertNotEqual(torch.tensor(n, device=device_type), n_astensor)


TestTensorCreation.test_as_tensor = _test_as_tensor


@expectedFailureMeta
@onlyNativeDeviceTypes
def _test_tensor_ctor_device_inference(self, device):
    torch_device = torch.device(device)
    values = torch.tensor((1, 2, 3), device=device)

    # Tests tensor and as_tensor
    # Note: warnings are suppressed (suppresses warnings)
    for op in (torch.tensor, torch.as_tensor):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(op(values).device, torch_device)
            self.assertEqual(op(values, dtype=torch.float64).device, torch_device)

            if self.device_type in ("cuda", "xpu"):
                with torch.get_device_module(self.device_type).device(device):
                    self.assertEqual(op(values.cpu()).device, torch.device("cpu"))

    # Tests sparse ctor
    indices = torch.tensor([[0, 1, 1], [2, 0, 1], [2, 1, 0]], device=device)
    sparse_size = (3, 3, 3)

    sparse_default = torch.sparse_coo_tensor(indices, values, sparse_size)
    self.assertEqual(sparse_default.device, torch_device)

    sparse_with_dtype = torch.sparse_coo_tensor(
        indices, values, sparse_size, dtype=torch.float64
    )
    self.assertEqual(sparse_with_dtype.device, torch_device)

    if self.device_type in ("cuda", "xpu"):
        with torch.get_device_module(self.device_type).device(device):
            sparse_with_dtype = torch.sparse_coo_tensor(
                indices.cpu(), values.cpu(), sparse_size, dtype=torch.float64
            )
            self.assertEqual(sparse_with_dtype.device, torch.device("cpu"))


TestTensorCreation.test_tensor_ctor_device_inference = (
    _test_tensor_ctor_device_inference
)


@onlyOn(["cuda", "xpu"])
@unittest.skipIf(
    IS_FBCODE or IS_SANDCASTLE, "Produces inconsistent errors when run in fbcode."
)
def _test_randperm_device_compatibility(self, device):
    device_type = torch.device(device).type
    gpu_gen = torch.Generator(device=device_type)
    cpu_gen = torch.Generator(device="cpu")

    # n=0 is a special case that we don't need to use generator, thus no error even if
    # device and generator don't match
    torch.randperm(
        0,
        device=f"{device_type}:0",
        generator=torch.Generator(device=f"{device_type}:0"),
    )
    if getattr(torch, device_type).device_count() > 1:
        torch.randperm(
            0,
            device=f"{device_type}:1",
            generator=torch.Generator(device=f"{device_type}:0"),
        )
    torch.randperm(0, device=device_type, generator=torch.Generator(device="cpu"))
    torch.randperm(0, device="cpu", generator=torch.Generator(device=device_type))

    for n in (1, 3, 100, 30000):
        torch.randperm(
            n, device=device_type, generator=torch.Generator(device=f"{device_type}:0")
        )
        torch.randperm(
            n, device=f"{device_type}:0", generator=torch.Generator(device=device_type)
        )
        # For gpu:0 to match gpu:1, we are making consistent device type matching
        # behavior just like torch.randint. Longer term, generator should ignore
        # device ordinal, since it's not used anyway.
        torch.randint(
            low=0,
            high=n + 1,
            size=(1,),
            device=f"{device_type}:0",
            generator=torch.Generator(device=f"{device_type}:1"),
        )
        torch.randperm(
            n,
            device=f"{device_type}:0",
            generator=torch.Generator(device=f"{device_type}:1"),
        )
        if getattr(torch, device_type).device_count() > 1:
            torch.randint(
                low=0,
                high=n + 1,
                size=(1,),
                device=f"{device_type}:1",
                generator=torch.Generator(device=f"{device_type}:0"),
            )
            torch.randperm(
                n,
                device=f"{device_type}:1",
                generator=torch.Generator(device=f"{device_type}:0"),
            )

        regex = "Expected a .* device type for generator but found .*"
        gpu_t = torch.tensor(n, device=device_type)
        self.assertRaisesRegex(
            RuntimeError,
            regex,
            lambda: torch.randperm(n, device=device_type, generator=cpu_gen),
        )
        self.assertRaisesRegex(
            RuntimeError,
            regex,
            lambda: torch.randperm(n, device=device_type, generator=cpu_gen, out=gpu_t),
        )
        cpu_t = torch.tensor(n, device="cpu")
        self.assertRaisesRegex(
            RuntimeError,
            regex,
            lambda: torch.randperm(n, device="cpu", generator=gpu_gen),
        )
        self.assertRaisesRegex(
            RuntimeError,
            regex,
            lambda: torch.randperm(n, device="cpu", generator=gpu_gen, out=cpu_t),
        )
        self.assertRaisesRegex(
            RuntimeError, regex, lambda: torch.randperm(n, generator=gpu_gen)
        )  # implicitly on CPU


TestRandomTensorCreation.test_randperm_device_compatibility = (
    _test_randperm_device_compatibility
)


# ======================================================================
# Add dtypesIfXPU overrides for migrated tensor-creation tests
# ======================================================================

TestTensorCreation.test_signal_window_functions = dtypesIfXPU(
    torch.float, torch.double, torch.bfloat16, torch.half, torch.long
)(TestTensorCreation.test_signal_window_functions)

TestTensorCreation.test_logspace_device_vs_cpu = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestTensorCreation.test_logspace_device_vs_cpu)

TestTensorCreation.test_logspace_base2 = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestTensorCreation.test_logspace_base2)

TestTensorCreation.test_logspace_special_steps = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(TestTensorCreation.test_logspace_special_steps)

TestTensorCreation.test_logspace = dtypesIfXPU(
    *all_types_and(torch.half, torch.bfloat16)
)(TestTensorCreation.test_logspace)

TestRandomTensorCreation.test_uniform_from_to = dtypesIfXPU(
    torch.float, torch.double, torch.half, torch.bfloat16
)(TestRandomTensorCreation.test_uniform_from_to)


# ======================================================================
# Add XPU largeTensorTest override for migrated randperm coverage
# ======================================================================

TestRandomTensorCreation.test_randperm_large = largeTensorTest("40GB", "xpu")(
    TestRandomTensorCreation.test_randperm_large
)


# ======================================================================
# Add XPU-only large tensor creation tests
# ======================================================================


@dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
@largeTensorTest(
    lambda self, device, dtype: (2**31) * torch.tensor([], dtype=dtype).element_size()
)
def _test_zeros_large(self, device, dtype):
    _ = torch.zeros(2**31 - 1, device=device, dtype=dtype)


TestLikeTensorCreation.test_zeros_large = _test_zeros_large


@dtypes(*all_types_and_complex_and(torch.half, torch.bool, torch.bfloat16))
@largeTensorTest(
    lambda self, device, dtype: (2**31) * torch.tensor([], dtype=dtype).element_size()
)
def _test_ones_large(self, device, dtype):
    _ = torch.ones(2**31 - 1, device=device, dtype=dtype)


TestLikeTensorCreation.test_ones_large = _test_ones_large


# ======================================================================
# Retarget outermost @onlyCUDA test methods to @onlyOn(["cuda", "xpu"])
# ======================================================================

TestTensorCreation.test_cat_channels_last_large_inputs = (
    retarget_outermost_onlycuda_to_onlyon(
        TestTensorCreation.test_cat_channels_last_large_inputs
    )
)

TestTensorCreation.test_cat_out_memory_format = retarget_outermost_onlycuda_to_onlyon(
    TestTensorCreation.test_cat_out_memory_format
)

TestTensorCreation.test_cat_stack_cross_devices = retarget_outermost_onlycuda_to_onlyon(
    TestTensorCreation.test_cat_stack_cross_devices
)

TestTensorCreation.test_cat = retarget_outermost_onlycuda_to_onlyon(
    TestTensorCreation.test_cat
)

TestTensorCreation.test_new_tensor_device = retarget_outermost_onlycuda_to_onlyon(
    TestTensorCreation.test_new_tensor_device
)

TestTensorCreation.test_range_factories_64bit_indexing = (
    retarget_outermost_onlycuda_to_onlyon(
        TestTensorCreation.test_range_factories_64bit_indexing
    )
)

TestAsArray.test_copy_from_tensor_mult_devices = retarget_outermost_onlycuda_to_onlyon(
    TestAsArray.test_copy_from_tensor_mult_devices
)

TestAsArray.test_copy_from_dlpack_mult_devices = retarget_outermost_onlycuda_to_onlyon(
    TestAsArray.test_copy_from_dlpack_mult_devices
)

TestAsArray.test_unsupported_alias_mult_devices = retarget_outermost_onlycuda_to_onlyon(
    TestAsArray.test_unsupported_alias_mult_devices
)


# ======================================================================
# Extend dtype coverage on XPU for the retargeted device-rounding test
# ======================================================================

TestTensorCreation.test_device_rounding = dtypesIfXPU(
    torch.half, torch.float, torch.double
)(retarget_outermost_onlycuda_to_onlyon(TestTensorCreation.test_device_rounding))


# ======================================================================
# Instantiate test classes for XPU execution
# ======================================================================

instantiate_device_type_tests(
    TestTensorCreation, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestRandomTensorCreation, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestLikeTensorCreation, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(TestBufferProtocol, globals(), only_for="cpu")
instantiate_device_type_tests(TestFromBlob, globals(), only_for="cpu")
instantiate_device_type_tests(TestAsArray, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
