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

import unittest

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._ops import DispatchKey
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyOn,
)
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfCrossRef,
    skipIfTorchDynamo,
    TEST_WITH_ASAN,
)
from torch.utils._pytree import tree_map

try:
    from xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx
except Exception:
    from .xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx

with XPUImportCtx(False):
    from test_decomp import (
        CROSS_REF_BACKWARD_EXCLUDE_SET,
        CROSS_REF_EXCLUDE_SET,
        DecompOneOffTests,
        decomposition_names,
        HasDecompTest,
        normalize_op_input_output,
        op_assert_ref_tol_table,
        ref_vjp_no_create,
        TestDecomp,
    )

# ======================================================================
# Tolerance table patches (XPU-specific entries)
# ======================================================================
# NOTE: When upstreaming this change, we need to provide a way to keep XPU-specific
# tolerances separate from the current CUDA entries.
op_assert_ref_tol_table.update(
    {
        (
            torch.float16,
            torch.ops.aten.nll_loss2d_backward.default,
        ): 1e-4,  # added (not in upstream)
        (
            torch.bfloat16,
            torch.ops.aten.reflection_pad2d_backward.default,
        ): 5e-2,  # upstream 5e-3 -> 5e-2
        (
            torch.float16,
            torch.ops.aten.log_sigmoid_backward.default,
        ): 2e-5,  # added (not in upstream)
    }
)


# ======================================================================
# CROSS_REF_EXCLUDE_SET patches (XPU-specific exclusions)
# ======================================================================

# XPU: max_pool2d_with_indices_backward tests are not applicable
# More details in https://github.com/pytorch/pytorch/pull/182619
CROSS_REF_EXCLUDE_SET.add(("xpu", None, "max_pool2d_with_indices_backward"))


# ======================================================================
# CROSS_REF_BACKWARD_EXCLUDE_SET patches (XPU-specific exclusions)
# ======================================================================

# XPU: max_pool backward tests are not applicable
# More details in https://github.com/pytorch/pytorch/pull/182619
CROSS_REF_BACKWARD_EXCLUDE_SET.update(
    {
        ("xpu", None, "nn.functional.max_pool1d"),
        ("xpu", None, "nn.functional.max_pool2d"),
    }
)


# ======================================================================
# core_backward_failures patches
# @skipOps eagerly bakes entries into fn._op_overrides at class
# definition time, so modifying the set after import has no effect.
# We patch _op_overrides on the method object directly.
# ======================================================================

_overrides = TestDecomp.test_quick_core_backward._op_overrides
# Un-skip logaddexp (passes on XPU).
# TODO: When integrating upstream, make the skip device-specific
# (i.e., skip("logaddexp", device_type="cuda")) so XPU is not affected.
_overrides.pop("logaddexp", None)


# ======================================================================
# Test overrides for XPU
# ======================================================================


# XPU adds (device_type, None, op.name) to test_keys for matching
# device-specific exclusion entries. Body otherwise matches upstream.
@skipIfTorchDynamo("Test does not work with TorchDynamo")
def _do_cross_ref(self, device, dtype, op, *, run_all):
    # XPU addition: (device_type, None, op.name) test_key for device-specific exclusions
    test_keys = [
        (torch.device(device).type, dtype, op.name),
        (
            torch.device(device).type,
            None,
            op.name,
        ),  # XPU: added for device-specific exclusions
        (None, dtype, op.name),
        (None, None, op.name),
    ]
    if any(key in CROSS_REF_EXCLUDE_SET for key in test_keys):
        self.skipTest(f"{op.name} in {dtype} not supported")

    skip_decomp_vjp = any(key in CROSS_REF_BACKWARD_EXCLUDE_SET for key in test_keys)

    requires_grad = (
        op.supports_autograd
        and dtype in op.supported_backward_dtypes(torch.device(device).type)
        # TODO: OpInfo really ought to error out for this case, but it's
        # not exercised in test_ops_gradients atm.  The problem is not
        # complex32 per-se (which is supported by data movement only ops)
        # but that when we do backwards we expect other ops like add to work
        and dtype != torch.complex32
    )
    samples = op.sample_inputs(device, dtype, requires_grad=requires_grad)

    aten_name = op.decomp_aten_name or op.aten_name

    func = op.get_op()

    def run_without_python_dispatcher(mode):
        return any(
            isinstance(op, torch._ops.OpOverload)
            and op.has_kernel_for_dispatch_key(DispatchKey.CompositeImplicitAutograd)
            for op in mode.decomposed.union([func])
        )

    for sample_input in samples:
        if requires_grad:
            fn, primals = normalize_op_input_output(func, sample_input)

            # Once https://github.com/pytorch/pytorch/pull/75965/ I can
            # store the called list on the mode object instance and no
            # explicit clearing is necessary as I will create a fresh mode
            # for each region
            with (
                self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all
                ) as mode,
                enable_python_dispatcher(),
            ):
                decomp_out, decomp_vjp_fn = ref_vjp_no_create(fn, *primals)
            if run_without_python_dispatcher(mode):
                # without this check, incorrect decomps at the python dispatcher level can still pass because
                # they're checking aten decomps at the torch_dispatch level.
                with self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all
                ) as mode:
                    decomp_out, decomp_vjp_fn = ref_vjp_no_create(fn, *primals)
            if aten_name in decomposition_names:
                self.check_decomposed(aten_name, mode)

            if not skip_decomp_vjp and (
                op.aten_backward_name in decomposition_names or run_all
            ):
                cotangents = tree_map(lambda x: torch.randn_like(x), decomp_out)

                with (
                    self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode,
                    enable_python_dispatcher(),
                ):
                    decomp_vjp_fn(cotangents)
                if run_without_python_dispatcher(mode):
                    # without this check, incorrect decomps at the python dispatcher level can still pass because
                    # they're checking aten decomps at the torch_dispatch level.
                    with self.DecompCrossRefMode(
                        self, self.precision, self.rel_tol, dtype, run_all
                    ) as mode:
                        decomp_vjp_fn(cotangents)
                if not run_all:
                    self.check_decomposed(op.aten_backward_name, mode)

        elif aten_name in decomposition_names or run_all:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            # A failure here might be because the decomposition for the op is wrong or because a
            # decomposition used by the particular op is wrong.
            with (
                self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all
                ) as mode,
                enable_python_dispatcher(),
            ):
                func(*args, **kwargs)

            if run_without_python_dispatcher(mode):
                # without this check, incorrect decomps at the python dispatcher level can still pass because
                # they're checking aten decomps at the torch_dispatch level.
                with self.DecompCrossRefMode(
                    self, self.precision, self.rel_tol, dtype, run_all
                ) as mode:
                    func(*args, **kwargs)

            if not run_all:
                self.check_decomposed(aten_name, mode)
        else:
            if not op.supports_autograd:
                raise AssertionError("expected op.supports_autograd")
            self.skipTest("only backwards is decomposed, but dtype doesn't support AD")


TestDecomp.do_cross_ref = _do_cross_ref


# Replaced @onlyCUDA decorator with @onlyOn(["cuda", "xpu"]).
# Replaced hardcoded device="cuda" with `device` argument from instantiate_device_type_tests.
@onlyOn(["cuda", "xpu"])
@unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
@skipIfCrossRef
def _test_amp_batch_norm_backward(self, device):
    grad_out = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
    x = torch.randn((1, 2, 16, 16), dtype=torch.float16, device=device)
    weight = torch.randn((2,), dtype=torch.float32, device=device)
    rmean = torch.randn((2,), dtype=torch.float32, device=device)
    rvar = torch.randn((2,), dtype=torch.float32, device=device)
    mean = torch.randn((0,), dtype=torch.float32, device=device)

    ref = torch.ops.aten.native_batch_norm_backward(
        grad_out,
        x,
        weight,
        rmean,
        rvar,
        mean,
        mean,
        False,
        1e-05,
        [True, True, True],
    )
    res = torch._decomp.decompositions.native_batch_norm_backward(
        grad_out,
        x,
        weight,
        rmean,
        rvar,
        mean,
        mean,
        False,
        1e-05,
        [True, True, True],
    )
    for a, b in zip(ref, res):
        self.assertEqual(a.stride(), b.stride())
        self.assertEqual(a.dtype, b.dtype)


DecompOneOffTests.test_amp_batch_norm_backward = _test_amp_batch_norm_backward


# Replaced @onlyCUDA decorator with @onlyOn(["cuda", "xpu"]).
# Converted @unittest.skipIf decorator into an in-body skip using self.skipTest
# so the device type is determined at runtime from the `device` argument rather
# than at decoration time from a module-level variable.
# Replaced hardcoded kernel name check with regex to work across backends (XPU/CUDA).
@onlyOn(["cuda", "xpu"])
def _test_rms_norm_decomp_gpu(self, device):
    if torch.device(device).type == "cuda" and not SM70OrLater:
        self.skipTest("triton requires CUDA with SM>=7.0")

    @torch.compile
    def rms_norm_sinh(a, b, c):
        output = torch.nn.functional.rms_norm(a, b, c)
        return torch.sinh(output)

    normalized_shape_arg = (3, 3, 3)
    input_tensor = torch.randn(3, 3, 3, device=device, requires_grad=True)
    weight_tensor = torch.randn(3, 3, 3, device=device, requires_grad=True)

    def forward_pass_fn():
        return rms_norm_sinh(input_tensor, normalized_shape_arg, weight_tensor)

    model_output, generated_codes = torch._inductor.utils.run_fw_bw_and_get_code(
        forward_pass_fn
    )

    # check RMSNorm stays fused with sinh/cosh in fw/bw kernels.
    # Kernel names can vary across backends (XPU/CUDA) and compiler versions.
    self.assertGreaterEqual(len(generated_codes), 2)
    self.assertRegex(generated_codes[0], r"triton_per_fused_.*rms_norm.*sinh")
    self.assertRegex(
        generated_codes[1], r"triton_per_fused_.*rms_norm.*backward.*cosh.*mul"
    )


# Renamed from test_rms_norm_decomp_cuda to test_rms_norm_decomp_gpu
DecompOneOffTests.test_rms_norm_decomp_gpu = _test_rms_norm_decomp_gpu

# Remove the upstream CUDA-only version
if hasattr(DecompOneOffTests, "test_rms_norm_decomp_cuda"):
    delattr(DecompOneOffTests, "test_rms_norm_decomp_cuda")


# ======================================================================
# Retargeting of CUDA-only tests to XPU
# ======================================================================

DecompOneOffTests.test_exponential_non_inf = retarget_outermost_onlycuda_to_onlyon(
    DecompOneOffTests.test_exponential_non_inf
)
DecompOneOffTests.test_fused_dropout_decomposition_extreme_p = (
    retarget_outermost_onlycuda_to_onlyon(
        DecompOneOffTests.test_fused_dropout_decomposition_extreme_p
    )
)
DecompOneOffTests.test_fused_dropout_compile_extreme_p = (
    retarget_outermost_onlycuda_to_onlyon(
        DecompOneOffTests.test_fused_dropout_compile_extreme_p
    )
)
DecompOneOffTests.test_addmm_out_dtype_decomp = retarget_outermost_onlycuda_to_onlyon(
    DecompOneOffTests.test_addmm_out_dtype_decomp
)


# ======================================================================
# Instantiate tests for XPU
# ======================================================================

instantiate_device_type_tests(TestDecomp, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(
    DecompOneOffTests, globals(), only_for="xpu", allow_xpu=True
)


if __name__ == "__main__":
    run_tests()
