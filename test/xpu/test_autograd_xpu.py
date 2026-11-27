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
from functools import partial

import torch
from torch.testing._internal.common_device_type import (
    dtypesIfXPU,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)
from torch.utils.flop_counter import FlopCounterMode

try:
    from xpu_test_utils import XPUImportCtx
except Exception:
    from .xpu_test_utils import XPUImportCtx


# Upstream: TestAutogradCudaOnly (only_for="cuda")
def forward_traceback_preserves_exception_with_checkpoint(self):
    with torch.library._scoped_library("_test_autograd", "FRAGMENT"):

        @torch.library.custom_op("_test_autograd::sin_op", mutates_args=())
        def sin_op(x: torch.Tensor) -> torch.Tensor:
            return x.sin()

        def setup_context(ctx, inputs, output):
            (x,) = inputs
            ctx.save_for_backward(x)

        def backward(ctx, grad):
            (x,) = ctx.saved_tensors
            return grad * x.cos()

        torch.library.register_autograd(
            "_test_autograd::sin_op",
            backward,
            setup_context=setup_context,
        )

        def fn(x):
            return torch.ops._test_autograd.sin_op(x)

        try:
            torch.xpu.memory._record_memory_history("all", stacks="python")
            x = torch.randn(4, device="xpu", requires_grad=True)
            y = checkpoint(fn, x, use_reentrant=False)
            y.sum().backward()
        finally:
            torch.xpu.memory._record_memory_history(None)


# Upstream: TestSelectiveActivationCheckpointCudaOnly (only_for="cuda")
def flops_and_mem(self):
    # From https://github.com/pytorch/pytorch/pull/126320
    def get_act_mem(f):
        out = f()
        out.backward()
        # Why do one forward and backward?
        start_mem = torch.xpu.memory_stats()["requested_bytes.all.current"]
        out = f()
        cur_mem = torch.xpu.memory_stats()["requested_bytes.all.current"]
        act_mem = (cur_mem - start_mem) / (1024 * 1024)
        out.backward()
        return act_mem

    def get_bw_flops(f):
        # Normalized so that a 512 square matmul returns 1
        f().backward()
        out = f()
        # NB: FlopCounterMode is pushed onto the mode stack before CachedMode, so
        # it will be able to observe whether an op is cached or not.
        with FlopCounterMode(display=False) as mode:
            out.backward()
        return mode.get_total_flops() / (512**3 * 2)

    x = torch.randn(512, 512, requires_grad=True, device="xpu")
    y = torch.randn(512, 512, requires_grad=True, device="xpu")

    def fn(x, y):
        return torch.mm(x.cos(), y).sin().sum()

    def fn_ac(x, y):
        return checkpoint(fn, x, y, use_reentrant=False)

    def fn_sac(x, y):
        context_fn = partial(
            create_selective_checkpoint_contexts,
            [
                torch.ops.aten.mm.default,
            ],
        )
        out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
        return out

    def policy_fn(ctx, op, *args, **kwargs):
        if op == torch.ops.aten.mm.default:
            return CheckpointPolicy.MUST_SAVE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE

    def fn_sac2(x, y):
        context_fn = partial(
            create_selective_checkpoint_contexts,
            policy_fn,
        )
        out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
        return out

    def policy_fn_bool(ctx, op, *args, **kwargs):
        return op == torch.ops.aten.mm.default

    def fn_sac3(x, y):
        context_fn = partial(
            create_selective_checkpoint_contexts,
            policy_fn_bool,
        )
        out = checkpoint(fn, x, y, use_reentrant=False, context_fn=context_fn)
        return out

    act_mem_noac = get_act_mem(lambda: fn(x, y))
    bw_flops_noac = get_bw_flops(lambda: fn(x, y))

    self.assertEqual(act_mem_noac, 2.0)
    self.assertEqual(bw_flops_noac, 2.0)

    act_mem_ac = get_act_mem(lambda: fn_ac(x, y))
    bw_flops_ac = get_bw_flops(lambda: fn_ac(x, y))

    self.assertEqual(act_mem_ac, 0.0)
    self.assertEqual(bw_flops_ac, 3.0)

    act_mem_sac = get_act_mem(lambda: fn_sac(x, y))
    bw_flops_sac = get_bw_flops(lambda: fn_sac(x, y))

    self.assertEqual(act_mem_sac, 1.0)
    self.assertEqual(bw_flops_sac, 2.0)

    act_mem_sac2 = get_act_mem(lambda: fn_sac2(x, y))
    bw_flops_sac2 = get_bw_flops(lambda: fn_sac2(x, y))

    self.assertEqual(act_mem_sac2, 1.0)
    self.assertEqual(bw_flops_sac2, 2.0)

    act_mem_sac3 = get_act_mem(lambda: fn_sac3(x, y))
    bw_flops_sac3 = get_bw_flops(lambda: fn_sac3(x, y))

    self.assertEqual(act_mem_sac3, 1.0)
    self.assertEqual(bw_flops_sac3, 2.0)


# torch.xpu has no default_stream(), which the upstream helper uses
def get_default_streams(self, num_devices=1):
    out = []
    for i in range(num_devices):
        with torch.accelerator.device_index(i):
            acc = torch.accelerator.current_accelerator()
            out.append(torch.get_device_module(acc).current_stream())
    return tuple(out)


# Upstream subprocess script hard-codes device="cuda"
def lookup_does_not_deadlock_with_python_dispatch(self, device):
    script = f"""
import sys
import threading

import torch
from torch.autograd import Function
from torch.utils._pytree import tree_map

dispatch_entered = threading.Event()
getter_entering_lookup = threading.Event()
sys.setswitchinterval(10)

class DispatchTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.shape,
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=False,
        )

    def __init__(self, elem):
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        dispatch_entered.set()
        if not getter_entering_lookup.wait(timeout=5):
            raise RuntimeError("timed out waiting for InputBuffer lookup")
        kwargs = {{}} if kwargs is None else kwargs
        unwrap = lambda value: value.elem if isinstance(value, cls) else value
        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

class Getter(Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        if not dispatch_entered.wait(timeout=5):
            raise RuntimeError("timed out waiting for Python dispatch")
        getter_entering_lookup.set()
        ctx.input_grad_buffers
        return grad_output

class Fanout(Function):
    @staticmethod
    def forward(ctx, getter_output, duplicated_a, duplicated_b):
        return duplicated_a.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output.cpu(),
            grad_output.clone(),
            DispatchTensor(grad_output.clone()),
        )

seed = torch.ones((), requires_grad=True)
getter_output = Getter.apply(seed)
source = torch.ones((), device="{device}", requires_grad=True)
duplicated = source.clone()
Fanout.apply(getter_output, duplicated, duplicated).backward()
"""
    try:
        subprocess.check_output(
            [sys.executable, "-c", script],
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.realpath(__file__)),
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        self.fail("input_grad_buffers lookup deadlocked")
    except subprocess.CalledProcessError as error:
        self.fail(error.output.decode("utf-8"))


with XPUImportCtx(False):
    from autograd.test_complex import TestAutogradComplex  # noqa: F401
    from autograd.test_logging import TestAutogradLogging  # noqa: F401
    from test_autograd import (
        TestAllowMutationOnSaved,
        TestAutograd,
        TestAutogradDeviceType,
        TestAutogradForwardMode,
        TestAutogradForwardModeBatchedGrad,
        TestAutogradInferenceMode,
        TestAutogradMultipleDispatch,
        TestAutogradStreamSynchronization,
        TestDisallowImplicitGradAccumulation,
        TestInputGradBuffers,
        TestMultithreadAutograd,  # noqa: F401
        TestMultithreadAutogradDevice,
        TestNestedCheckpoint,
        TestSACAmbientSavedTensorsHooksDeviceType,
        TestSelectiveActivationCheckpoint,
    )

TestAutograd.test_forward_traceback_preserves_exception_with_checkpoint = (
    forward_traceback_preserves_exception_with_checkpoint
)
TestSelectiveActivationCheckpoint.test_flops_and_mem = flops_and_mem
TestAutogradStreamSynchronization.get_default_streams = get_default_streams
TestInputGradBuffers.test_lookup_does_not_deadlock_with_python_dispatch = (
    lookup_does_not_deadlock_with_python_dispatch
)
# Upstream enables float16 only via dtypesIfCUDA
TestAutogradDeviceType.test_set_requires_grad_only_for_floats = dtypesIfXPU(
    torch.half,
    torch.float,
    torch.double,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)(TestAutogradDeviceType.test_set_requires_grad_only_for_floats)

instantiate_device_type_tests(
    TestAutogradDeviceType, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestAutogradMultipleDispatch, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestAutogradStreamSynchronization, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestMultithreadAutogradDevice, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestSACAmbientSavedTensorsHooksDeviceType,
    globals(),
    only_for="xpu",
    allow_xpu=True,
)
instantiate_device_type_tests(
    TestInputGradBuffers, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestDisallowImplicitGradAccumulation, globals(), only_for="xpu", allow_xpu=True
)

instantiate_parametrized_tests(TestAutograd)
instantiate_parametrized_tests(TestNestedCheckpoint)
instantiate_parametrized_tests(TestAllowMutationOnSaved)
instantiate_parametrized_tests(TestAutogradForwardMode)
instantiate_parametrized_tests(TestAutogradForwardModeBatchedGrad)
instantiate_parametrized_tests(TestAutogradInferenceMode)


if __name__ == "__main__":
    run_tests()
