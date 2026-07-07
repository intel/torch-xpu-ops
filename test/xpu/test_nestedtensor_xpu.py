# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# ruff: noqa: F401

# Owner(s): ["module: intel"]

import inspect

import torch
from torch.testing._internal.common_cuda import SM70OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfXPU,
    instantiate_device_type_tests,
    onlyOn,
    skipCUDAIf,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
    serialTest,
    skipIfXpu,
    xfailIfTorchDynamo,
)

try:
    from xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx
except Exception:
    from .xpu_test_utils import retarget_outermost_onlycuda_to_onlyon, XPUImportCtx

with XPUImportCtx(False):
    from test_nestedtensor import (
        convert_jagged_to_nested_tensor,
        convert_nt_to_jagged,
        random_nt_noncontiguous_pair,
        TestNestedInt,
        TestNestedTensor,
        TestNestedTensorAutograd,
        TestNestedTensorDeviceType,
        TestNestedTensorOpInfo,
        TestNestedTensorSubclass,
    )


# ======================================================================
# Retarget outermost @onlyCUDA test methods to @onlyOn(["cuda", "xpu"])
# ======================================================================

TestNestedTensorDeviceType.test_nested_tensor_dense_elementwise = (
    retarget_outermost_onlycuda_to_onlyon(
        TestNestedTensorDeviceType.test_nested_tensor_dense_elementwise
    )
)
TestNestedTensorDeviceType.test_bmm_cuda = retarget_outermost_onlycuda_to_onlyon(
    TestNestedTensorDeviceType.test_bmm_cuda
)
TestNestedTensorSubclass.test_noncontiguous_to = retarget_outermost_onlycuda_to_onlyon(
    TestNestedTensorSubclass.test_noncontiguous_to
)

# ======================================================================
# dtypesIfXPU decorator additions
# ======================================================================

TestNestedTensorDeviceType.test_layer_norm = dtypesIfXPU(torch.float, torch.half)(
    TestNestedTensorDeviceType.test_layer_norm
)
TestNestedTensorDeviceType.test_layer_norm_breaking = dtypesIfXPU(
    torch.float, torch.half
)(TestNestedTensorDeviceType.test_layer_norm_breaking)


# ======================================================================
# Method overrides for XPU-specific behavior
# ======================================================================


@onlyOn(["cuda", "xpu"])
@dtypes(torch.float32)
@serialTest()
def _test_linear_backward_memory_usage(self, device, dtype):
    B, D, max_seq_len = 64, 512, 100
    if torch.device(device).type == "xpu":
        reset_peak_memory_stats = torch.xpu.reset_peak_memory_stats
        max_memory_allocated = torch.xpu.max_memory_allocated
    else:
        torch._C._cuda_clearCublasWorkspaces()
        reset_peak_memory_stats = torch.cuda.reset_max_memory_allocated
        max_memory_allocated = torch.cuda.max_memory_allocated

    m = torch.nn.Linear(D, D, device=device)
    nt = torch.nested.as_nested_tensor(
        [
            torch.rand(size=[seq_len, D])
            for seq_len in torch.randint(max_seq_len, size=(B,))
        ],
        layout=torch.jagged,
        device=device,
    )

    nt = nt.unsqueeze(-2)
    reset_peak_memory_stats()
    m(nt).sum().backward()
    max_after_gb = max_memory_allocated(0) // (1024**3)
    self.assertEqual(max_after_gb, 0)


TestNestedTensorSubclass.test_linear_backward_memory_usage = (
    _test_linear_backward_memory_usage
)


TestNestedTensorSubclass.test_pin_memory = onlyOn(["cuda", "xpu"])(
    inspect.unwrap(TestNestedTensorSubclass.test_pin_memory)
)


@xfailIfTorchDynamo
@onlyOn(["cuda", "xpu"])
def _test_jagged_layout_construction_with_pinned_memory(self, device):
    for tensor_list in self._get_example_tensor_lists():
        nt = torch.nested.nested_tensor(
            tensor_list, layout=torch.jagged, device="cpu", pin_memory=True
        )

        expected_dim = torch.as_tensor(tensor_list[0]).dim() + 1
        expected_batch_size = len(tensor_list)
        expected_min_seqlen = min(
            (torch.tensor(t) if isinstance(t, list) else t).shape[0]
            for t in tensor_list
        )
        expected_max_seqlen = max(
            (torch.tensor(t) if isinstance(t, list) else t).shape[0]
            for t in tensor_list
        )
        self._validate_nt(
            nt,
            device="cpu",
            dtype=torch.float32,
            layout=torch.jagged,
            requires_grad=False,
            dim=expected_dim,
            batch_size=expected_batch_size,
            contiguous=True,
            cached_min_seqlen=expected_min_seqlen,
            cached_max_seqlen=expected_max_seqlen,
        )
        self.assertTrue(nt.is_pinned())


TestNestedTensorSubclass.test_jagged_layout_construction_with_pinned_memory = (
    _test_jagged_layout_construction_with_pinned_memory
)


@onlyOn(["cuda", "xpu"])
@dtypes(torch.double, torch.half)
def _test_device_dtype_transfer_updates_offsets(self, device, dtype):
    for tensor_list in self._get_example_tensor_lists():
        orig_device = torch.device("cpu")
        orig_dtype = torch.float32
        nt = torch.nested.nested_tensor(
            tensor_list, layout=torch.jagged, device=orig_device, dtype=orig_dtype
        )

        self.assertEqual(torch.int64, nt.offsets().dtype)
        nt = nt.to(device=device).to(dtype=dtype)

        self.assertEqual(nt.values().device, nt.offsets().device)
        self.assertEqual(torch.int64, nt.offsets().dtype)


TestNestedTensorSubclass.test_device_dtype_transfer_updates_offsets = (
    _test_device_dtype_transfer_updates_offsets
)


# Mirror upstream's decorator stack; only onlyCUDA -> onlyOn and add XPU dtypes.
TestNestedTensorSubclass.test_sdpa_backwards = skipCUDAIf(
    not SM70OrLater, "GPU capability is < SM70"
)(
    onlyOn(["cuda", "xpu"])(
        dtypesIfXPU(torch.bfloat16)(
            inspect.unwrap(TestNestedTensorSubclass.test_sdpa_backwards)
        )
    )
)


# ======================================================================
# SDPA dtype overrides and XPU skips
# ======================================================================

TestNestedTensorSubclass.test_record_stream = skipIfXpu(
    msg="test doesn't currently work on the XPU stack"
)(TestNestedTensorSubclass.test_record_stream)
TestNestedTensorSubclass.test_sdpa = dtypesIfXPU(torch.bfloat16)(
    TestNestedTensorSubclass.test_sdpa
)
_test_sdpa_compile = TestNestedTensorSubclass.test_sdpa_compile


@dtypes(torch.bfloat16)
def _test_sdpa_compile_xpu(self, device, dtype):
    bypass_device_restrictions = self.bypass_device_restrictions
    self.bypass_device_restrictions = True
    try:
        return _test_sdpa_compile(self, device, dtype)
    finally:
        self.bypass_device_restrictions = bypass_device_restrictions


TestNestedTensorSubclass.test_sdpa_compile = dtypesIfXPU(torch.bfloat16)(
    _test_sdpa_compile_xpu
)
TestNestedTensorSubclass.test_sdpa_with_packed_in_proj = dtypesIfXPU(
    torch.float16, torch.bfloat16, torch.float32
)(
    skipIfXpu(msg="XPU does not support the NestedTensor SDPA packed in-proj path.")(
        TestNestedTensorSubclass.test_sdpa_with_packed_in_proj
    )
)
TestNestedTensorSubclass.test_sdpa_flop_counter = skipIfXpu(
    msg="XPU does not support NestedTensor SDPA flop-counter coverage in this test."
)(TestNestedTensorSubclass.test_sdpa_flop_counter)
TestNestedTensorSubclass.test_dummy_mha_with_nt = skipIfXpu(
    msg="XPU does not support NestedTensor for SDPA operations."
)(TestNestedTensorSubclass.test_dummy_mha_with_nt)


# ======================================================================
# Instantiate test classes for XPU execution
# ======================================================================

instantiate_parametrized_tests(TestNestedTensor)
instantiate_device_type_tests(
    TestNestedTensorDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestNestedTensorAutograd, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestNestedTensorSubclass, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestNestedTensorOpInfo, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
