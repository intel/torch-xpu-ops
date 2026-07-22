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
# ruff: noqa: F401

import copy
import itertools
from functools import partial

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyOn,
    OpDTypes,
    ops,
    skip,
    xfail,
)
from torch.testing._internal.common_dtype import (
    complex_types_and,
    integral_types,
    integral_types_and,
)
from torch.testing._internal.common_methods_invocations import binary_ufuncs, op_db
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfCrossRef,
    suppress_warnings,
)
from torch.testing._internal.opinfo.core import S, SampleInput

try:
    from xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx
except Exception:
    from .xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx

bf16 = torch.bfloat16
f64 = torch.float64
f16 = torch.float16
c32 = torch.complex32
c64 = torch.complex64
c128 = torch.complex128
b8 = torch.bool

# Patch bf16 dtype coverage for ops that are missing it on XPU.
_ops_missing_bf16 = [
    "addbmm",
    "__rmatmul__",
    "bmm",
    "matmul",
    "nn.functional.bilinear",
    "torch.ops.aten._efficient_attention_forward",
]
_ops_missing_bf16_expected = set(_ops_missing_bf16)
_ops_missing_bf16_updated = set()
_ops_count = 0
for _op in op_db:
    if _op.name in _ops_missing_bf16_expected:
        _ops_missing_bf16_updated.add(_op.name)
        _ops_count += 1
        for _dtype_list in [_op.dtypesIfCUDA, _op.dtypesIfXPU, _op.dtypesIf.get("xpu")]:
            if _dtype_list is not None and bf16 not in _dtype_list:
                _dtype_list.add(bf16)
        if _ops_count == len(_ops_missing_bf16_expected):
            break
_ops_not_found = _ops_missing_bf16_expected - _ops_missing_bf16_updated
assert not _ops_not_found, (
    f"Failed to update bf16 dtype coverage for expected ops in op_db: "
    f"{sorted(_ops_not_found)}. Verify op names exist in operator database."
)


# NOTE: only needed in this wrapper - in upstream use the original function
import unittest


def skipOps(to_skip):
    def wrapped(fn):
        from torch.testing._internal.opinfo.core import DecorateInfo

        parts = fn.__qualname__.split(".")
        test_name = parts[-1].lstrip("_")
        cls_name = parts[-2] if len(parts) >= 2 else ""
        overrides = getattr(fn, "_op_overrides", {})
        for skip_spec in to_skip:
            if hasattr(skip_spec, "op_name"):
                op_name = skip_spec.op_name
                variant_name = skip_spec.variant_name
                device_type = skip_spec.device_type
                dtypes = skip_spec.dtypes
                if hasattr(skip_spec, "decorator"):
                    decorator_callable = skip_spec.decorator
                else:
                    expected_failure = skip_spec.expected_failure
                    decorator_callable = (
                        unittest.expectedFailure
                        if expected_failure
                        else unittest.skip("Skipped!")
                    )
            else:
                op_name, variant_name, device_type, dtypes, expected_failure = skip_spec
                decorator_callable = (
                    unittest.expectedFailure
                    if expected_failure
                    else unittest.skip("Skipped!")
                )
            full_name = f"{op_name}.{variant_name}" if variant_name else op_name

            decorator = DecorateInfo(
                decorator_callable,
                None,  # cls_name=None to match any class (wrapper functions are module-level, not inside class)
                test_name,
                device_type=device_type,
                dtypes=dtypes,
            )
            overrides.setdefault(full_name, []).append(decorator)
        fn._op_overrides = overrides
        return fn

    return wrapped


with XPUImportCtx(False):
    from test_meta import (
        aten,
        CHECK_STRIDES,
        CHECK_STRIDES_SKIPS,
        foreach_op_db,
        get_strided_args,
        meta_dispatch_device_expected_failures,
        meta_dispatch_device_skips,
        meta_dispatch_expected_failures,
        meta_dispatch_skips,
        meta_function_device_expected_failures,
        meta_function_device_skips,
        meta_function_expected_failures,
        MetaConverter,
        TestMeta,
        TestMetaConverter,
    )

# ======================================================================
# Module-level variable patches
# ======================================================================

# Move _fft entries from CHECK_STRIDES to CHECK_STRIDES_SKIPS on XPU
# (XPU stride behavior for FFT ops doesn't match CUDA reference)
_fft_ops = {aten._fft_c2c.default, aten._fft_c2r.default, aten._fft_r2c.default}
CHECK_STRIDES -= _fft_ops
CHECK_STRIDES_SKIPS |= _fft_ops

meta_function_device_expected_failures["xpu"] = {
    **meta_function_device_expected_failures["cuda"],
}
meta_function_device_skips["xpu"] = {
    **meta_function_device_skips["cuda"],
}

meta_function_device_skips["xpu"].update(
    {
        torch.fft.fft: {f16},
        torch.fft.ifft: {f16},
        torch.fft.rfft: {f16},
        torch.fft.ihfft: {f16},
        torch.fft.fft2: {f16},
        torch.fft.ifft2: {f16},
        torch.fft.rfft2: {f16},
        torch.fft.ihfft2: {f16},
        torch.fft.fftn: {f16},
        torch.fft.ifftn: {f16},
        torch.fft.rfftn: {f16},
        torch.fft.ihfftn: {f16},
        torch.fft.hfft: {f16},
        torch.fft.irfft: {f16},
        torch.fft.irfft2: {f16},
        torch.fft.hfft2: {f16},
        torch.fft.irfftn: {f16},
        torch.fft.hfftn: {f16},
    }
)

meta_dispatch_device_skips["xpu"] = meta_dispatch_device_skips["cuda"]
meta_dispatch_device_expected_failures["xpu"] = meta_dispatch_device_expected_failures[
    "cuda"
]

# ======================================================================
# Override @onlyCUDA test methods to also run on XPU
# ======================================================================


# Removed @onlyCUDA; replaced with @onlyOn(["cuda", "xpu"]).
@skipIfCrossRef
@suppress_warnings
@skipOps(
    (
        xfail("view_as_complex"),
        skip("sparse.sampled_addmm"),
        xfail("nn.functional.binary_cross_entropy"),
        xfail("narrow_copy"),
        xfail("view_copy"),
        xfail("view"),
        xfail("view_as"),
        xfail("empty_strided"),
        skip(
            "normal", device_type="cuda"
        ),  # TODO: this test passes on XPU, when upstreaming set skip for CUDA only
        xfail("take_along_dim"),
        xfail("kron"),
        xfail("nn.functional.channel_shuffle"),
        xfail("_foreach_sub"),
        xfail("_foreach_clamp_min", dtypes=complex_types_and(b8)),
        xfail("_foreach_clamp_max", dtypes=complex_types_and(b8)),
        xfail("_foreach_minimum", dtypes=complex_types_and(b8)),
        xfail("_foreach_maximum", dtypes=complex_types_and(b8)),
        xfail("_foreach_addcmul", dtypes=(b8,)),
        xfail("_foreach_addcdiv", dtypes=integral_types() + complex_types_and(b8)),
        xfail("_foreach_lerp", dtypes=integral_types_and(b8)),
        xfail(
            "native_group_norm", device_type="xpu"
        ),  # TODO: move to common_method_invocations.py when upstreaming
    )
)
@ops(itertools.chain(op_db, foreach_op_db), dtypes=OpDTypes.any_common_cpu_cuda_one)
@onlyOn(["cuda", "xpu"])
def _test_dispatch_symbolic_meta_outplace_all_strides(self, device, dtype, op):
    self._run_dispatch_meta_test(
        device, dtype, op, symbolic_meta=True, inplace=False, all_stride_variants=True
    )


# Keep the upstream test_... name for DecorateInfo/opinfo matching, while using
# a private helper name to prevent pytest from collecting it as a module-level test.
_test_dispatch_symbolic_meta_outplace_all_strides.__name__ = (
    "test_dispatch_symbolic_meta_outplace_all_strides"
)
TestMeta.test_dispatch_symbolic_meta_outplace_all_strides = (
    _test_dispatch_symbolic_meta_outplace_all_strides
)


# Removed @onlyCUDA; replaced with @onlyOn(["cuda", "xpu"]).
@skipIfCrossRef
@suppress_warnings
@skipOps(
    (
        xfail("abs", dtypes=(c128, c64, c32)),
        xfail("as_strided", variant_name="partial_views"),
        xfail(
            "_foreach_add",
            dtypes=integral_types() + complex_types_and(b8, bf16, f16, f64),
        ),
        xfail("_foreach_sub"),
        xfail("_foreach_mul", dtypes=(b8,)),
        xfail("_foreach_div", dtypes=integral_types_and(b8)),
        xfail("_foreach_clamp_min", dtypes=complex_types_and(b8)),
        xfail("_foreach_clamp_max", dtypes=complex_types_and(b8)),
        xfail("_foreach_minimum", dtypes=complex_types_and(b8)),
        xfail("_foreach_maximum", dtypes=complex_types_and(b8)),
        xfail("_foreach_addcmul", dtypes=integral_types() + complex_types_and(b8)),
        xfail("_foreach_addcdiv", dtypes=integral_types() + complex_types_and(b8)),
        xfail("_foreach_norm"),
        xfail("_foreach_lerp", dtypes=integral_types_and(b8)),
    )
)
@ops(itertools.chain(op_db, foreach_op_db), dtypes=OpDTypes.any_common_cpu_cuda_one)
@onlyOn(["cuda", "xpu"])
def _test_dispatch_symbolic_meta_inplace_all_strides(self, device, dtype, op):
    self._run_dispatch_meta_test(
        device, dtype, op, symbolic_meta=True, inplace=True, all_stride_variants=True
    )


_test_dispatch_symbolic_meta_inplace_all_strides.__name__ = (
    "test_dispatch_symbolic_meta_inplace_all_strides"
)
TestMeta.test_dispatch_symbolic_meta_inplace_all_strides = (
    _test_dispatch_symbolic_meta_inplace_all_strides
)


# Removed @onlyCUDA; replaced with @onlyOn(["cuda", "xpu"]).
@skipIfCrossRef
@suppress_warnings
@skipOps(
    (
        xfail("complex"),
        xfail("heaviside"),
        xfail("isclose"),
        xfail("polar"),
        xfail("_refs.copysign"),
        xfail("_refs.floor_divide"),
        xfail("_refs.isclose"),
        xfail("_refs._conversions.complex"),
        xfail("_refs._conversions.polar"),
    )
)
@ops(binary_ufuncs, allowed_dtypes=(torch.float32,))
@onlyOn(["cuda", "xpu"])
def _test_binary_ufuncs_mixed_dtype(self, device, dtype, op):
    make_arg = partial(
        make_tensor,
        device=device,
    )

    def sample_input(op, device, dtype, requires_grad, **kwargs):
        yield SampleInput(
            make_arg((S,), dtype=dtype), make_arg((S,), dtype=torch.float16)
        )

    op = copy.copy(op)
    op.sample_inputs_func = sample_input

    self._run_dispatch_meta_test(device, dtype, op, symbolic_meta=True, inplace=False)


_test_binary_ufuncs_mixed_dtype.__name__ = "test_binary_ufuncs_mixed_dtype"
TestMeta.test_binary_ufuncs_mixed_dtype = _test_binary_ufuncs_mixed_dtype


# Removed @onlyCUDA; replaced with @onlyOn(["cuda", "xpu"]).
# test_fill_stride has no `device` parameter — it's a simple TestCase method,
# not a device-type parameterized test.
TestMeta.test_fill_stride = retarget_outermost_onlycuda_to_onlyon(
    TestMeta.test_fill_stride
)


# ======================================================================
# 3. Override print_op_str_if_not_supported to use dynamic device_type
# ======================================================================

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


def print_op_str_if_not_supported(op_str):
    from torchgen.model import OperatorName

    op = OperatorName.parse(op_str)
    packet = getattr(torch.ops.aten, str(op.name))
    overload = getattr(packet, op.overload_name if op.overload_name else "default")
    if any(
        overload in d
        for d in [meta_dispatch_skips, meta_dispatch_device_skips[device_type]]
    ):
        print(f"{overload}  # SKIP")
    if any(
        overload in d
        for d in [
            meta_dispatch_expected_failures,
            meta_dispatch_device_expected_failures[device_type],
        ]
    ):
        print(overload)


# ======================================================================
# 4. Instantiate tests for XPU
# ======================================================================

instantiate_device_type_tests(TestMeta, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    import os
    import sys

    import yaml
    from torchgen.yaml_utils import YamlLoader

    COMPARE_XLA = os.getenv("PYTORCH_COMPARE_XLA", None)
    if COMPARE_XLA is not None:
        with open(COMPARE_XLA) as f:
            d = yaml.load(f, Loader=YamlLoader)
            ops = (
                d.get("full_codegen", [])
                + d.get("supported", [])
                + d.get("autograd", [])
            )
            for op_str in ops:
                print_op_str_if_not_supported(op_str)
        sys.exit(0)

    COMPARE_TEXT = os.getenv("PYTORCH_COMPARE_TEXT", None)
    if COMPARE_TEXT is not None:
        with open(COMPARE_TEXT) as f:
            for op_str in f:
                print_op_str_if_not_supported(op_str.strip())
        sys.exit(0)

    run_tests()
