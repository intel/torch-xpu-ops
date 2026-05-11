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

import functools
from contextlib import nullcontext
from unittest.mock import patch as mock_patch

import torch
import torch._dynamo
import torch._functorch
import torch._functorch.config
from torch._dynamo.backends.common import aot_autograd
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_XPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_xpu_and_triton

# =============================================================================
# XPU monkey-patches for functionalize_rng_ops support
#
# These patches extend PyTorch's RNG functionalization to support XPU devices.
# The upstream PyTorch implementations of CUDARngStateHelper and philox_rand
# currently only support CUDA.  These patches add equivalent XPU support and
# are required until the corresponding upstream PyTorch fixes land.
#
# Upstream tracking: https://github.com/pytorch/pytorch/pull/174370
#
# Files patched (upstream):
#   torch/_prims_common/__init__.py        - CUDARngStateHelper
#   torch/_decomp/decompositions_for_rng.py - rand / rand_like decompositions
#   torch/_prims/rng_prims.py              - philox_rand implementation
#   torch/_functorch/_aot_autograd/
#       graph_capture_wrappers.py          - XPU RNG state patching in traces
# =============================================================================

# Keep alive any Library objects we register (gc'd objects deregister kernels)
_xpu_rng_libs: list = []


def _apply_xpu_rng_patches() -> None:
    """
    Apply all XPU RNG monkey-patches required for
    ``functionalize_rng_ops=True`` to work on XPU devices.
    """
    import torch._prims_common as prims_common
    from torch._decomp import register_decomposition as _reg_decomp
    from torch._decomp.decompositions_for_rng import (
        PhiloxStateTracker,
        rng_decompositions,
    )

    # ------------------------------------------------------------------
    # 1.  CUDARngStateHelper: support XPU when CUDA is not available
    # ------------------------------------------------------------------

    @staticmethod  # type: ignore[misc]
    def _get_torch_state_as_tuple_xpu(fake_mode=nullcontext()):
        if torch.cuda.is_available():
            with fake_mode:
                seed = torch.tensor(torch.cuda.initial_seed())
                offset = torch.tensor(torch.cuda._get_rng_state_offset())
                return seed, offset
        elif torch.xpu.is_available():
            with fake_mode:
                seed = torch.tensor(torch.xpu.initial_seed())
                offset = torch.tensor(torch.xpu._get_rng_state_offset())
                return seed, offset
        raise RuntimeError(
            "functionalize_rng_ops requires either CUDA or XPU to be available"
        )

    @staticmethod  # type: ignore[misc]
    def _set_torch_state_tensor_xpu(seed, offset):
        seed_portion = seed.reshape([1]).view(torch.uint8)
        offset_portion = offset.reshape([1]).view(torch.uint8)
        new_state = torch.cat([seed_portion, offset_portion])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(new_state)
        else:
            torch.xpu.set_rng_state(new_state)

    @staticmethod  # type: ignore[misc]
    def _set_new_offset_xpu(relative_offset):
        if torch.cuda.is_available():
            torch.cuda._set_rng_state_offset(relative_offset.item())
        else:
            torch.xpu._set_rng_state_offset(relative_offset.item())

    prims_common.CUDARngStateHelper.get_torch_state_as_tuple = (
        _get_torch_state_as_tuple_xpu
    )
    prims_common.CUDARngStateHelper.set_torch_state_tensor = _set_torch_state_tensor_xpu
    prims_common.CUDARngStateHelper.set_new_offset = _set_new_offset_xpu

    # ------------------------------------------------------------------
    # 2.  decompositions_for_rng: extend rand / rand_like to accept XPU
    #
    #     We re-register into rng_decompositions so that AOTAutograd's
    #     functionalize_rng_ops code path uses our XPU-aware versions.
    # ------------------------------------------------------------------

    @_reg_decomp([torch.ops.aten.rand], rng_decompositions)  # type: ignore[arg-type]
    def _rand_xpu(
        shape, dtype=None, layout=torch.strided, device=None, pin_memory=False
    ):
        if device and device.type not in ("cuda", "xpu"):
            raise RuntimeError(
                f"You are trying to functionalize a {device.type} RNG operator "
                f"but {device.type} does not use Philox/counter-based RNG."
            )
        seed, offset = PhiloxStateTracker.get_state_as_tuple()
        dtype = dtype or torch.float32
        out, offset_jump = torch.ops.rngprims.philox_rand(
            shape, seed, offset, None, device, dtype
        )
        PhiloxStateTracker.advance_offset(offset_jump)
        return out

    @_reg_decomp([torch.ops.aten.rand_like], rng_decompositions)  # type: ignore[arg-type]
    def _rand_like_xpu(
        x,
        dtype=None,
        layout=None,
        device=None,
        pin_memory=False,
        memory_format=torch.preserve_format,
    ):
        device = device or x.device
        if device.type not in ("cuda", "xpu"):
            raise RuntimeError(
                f"You are trying to functionalize a {device.type} RNG operator "
                f"but {device.type} does not use Philox/counter-based RNG."
            )
        dtype = dtype or x.dtype
        seed, offset = PhiloxStateTracker.get_state_as_tuple()
        out, offset_jump = torch.ops.rngprims.philox_rand(
            x.shape, seed, offset, None, device, dtype
        )
        PhiloxStateTracker.advance_offset(offset_jump)
        return out

    # ------------------------------------------------------------------
    # 3.  philox_rand CPU kernel: handle XPU as target device
    #
    #     seed / offset are always CPU tensors, so philox_rand dispatches
    #     through the CPU key.  We override the CPU kernel so that when
    #     the target *device* argument is "xpu" we use XPU RNG APIs.
    # ------------------------------------------------------------------

    rng_lib = torch.library.Library("rngprims", "IMPL")  # noqa: TOR901

    def _philox_rand_cpu_with_xpu(shape, seed, offset, stride, device, dtype):
        if stride is not None:
            raise AssertionError(f"stride must be None, got {stride}")

        if device is None:
            device = torch.device("cpu")
        if isinstance(device, str):
            device = torch.device(device)

        if device.type == "cuda":
            # Original CUDA path: fork RNG state, set seed+offset, generate
            with torch.random.fork_rng([device]):
                seed_portion = seed.reshape([1]).view(torch.uint8)
                offset_portion = offset.reshape([1]).view(torch.uint8)
                torch.cuda.set_rng_state(torch.cat([seed_portion, offset_portion]))
                random_values = torch.rand(shape, device=device, dtype=dtype)
                new_offset = torch.cuda._get_rng_state_offset()
            offset_jump = new_offset - int(offset.item())
            return random_values, torch.tensor(offset_jump)

        elif device.type == "xpu":
            # XPU path: fork XPU RNG state, set seed+offset, generate
            dev_idx = device.index if device.index is not None else 0
            xpu_device = torch.device("xpu", dev_idx)
            with torch.random.fork_rng([xpu_device], device_type="xpu"):
                torch.xpu.manual_seed(int(seed.item()))
                torch.xpu._set_rng_state_offset(int(offset.item()), device)
                random_values = torch.rand(shape, device=device, dtype=dtype)
                new_offset = torch.xpu._get_rng_state_offset(device)
            offset_jump = new_offset - int(offset.item())
            return random_values, torch.tensor(offset_jump)

        raise RuntimeError(
            f"You are trying to functionalize a {device.type} RNG operator "
            f"but {device.type} does not use Philox/counter-based RNG."
        )

    rng_lib.impl("philox_rand", _philox_rand_cpu_with_xpu, "CPU")
    # Keep the Library alive; GC-ing it deregisters the kernel
    _xpu_rng_libs.append(rng_lib)

    # ------------------------------------------------------------------
    # 4.  graph_capture_wrappers: patch XPU RNG state inside traced fns
    #
    #     The original create_functionalized_rng_ops_wrapper patches
    #     torch.cuda.get/set_rng_state.  We extend it to also patch the
    #     XPU equivalents so they route through PhiloxStateTracker.
    # ------------------------------------------------------------------
    import torch._functorch._aot_autograd.graph_capture_wrappers as gcw

    _orig_create_wrapper = gcw.create_functionalized_rng_ops_wrapper
    _PST = PhiloxStateTracker  # alias for use inside closure

    def _create_wrapper_xpu(func, args, args_descs, trace_joint=True):
        def _override_get(device=None):
            return _PST.get_state_as_tensor()

        def _override_set(x, device=None):
            _PST.set_state_from_tensor(x)

        original_func = func

        def _xpu_wrapped(*a, **kw):
            with (
                mock_patch("torch.xpu.get_rng_state", _override_get),
                mock_patch("torch.xpu.set_rng_state", _override_set),
            ):
                return original_func(*a, **kw)

        return _orig_create_wrapper(_xpu_wrapped, args, args_descs, trace_joint)

    gcw.create_functionalized_rng_ops_wrapper = _create_wrapper_xpu


# Apply patches at module-import time (only when XPU is present)
if torch.xpu.is_available():
    _apply_xpu_rng_patches()


# =============================================================================
# Helpers (mirrors the count_ops helper in upstream test_higher_order_ops.py)
# =============================================================================


def count_ops(gm, args, freq, op):
    actual = [node.target for node in gm.graph.nodes].count(op)
    if actual != freq:
        raise AssertionError(f"expected={freq}, actual={actual}")
    return gm


# =============================================================================
# Test class
# =============================================================================


class ActivationCheckpointingTests(TestCase):
    """
    XPU adaptation of upstream PyTorch's ActivationCheckpointingTests from
    ``test/dynamo/test_higher_order_ops.py``.

    Changes vs. upstream:
    * ``@requires_cuda_and_triton``  →  ``@requires_xpu_and_triton``
    * ``device="cuda"``              →  ``device="xpu"``
    * RNG helpers patched above to route CUDA calls to XPU equivalents.

    Upstream test reference:
      **test/dynamo/test_higher_order_ops.py::ActivationCheckpointingTests::test_dropout**
    """

    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        cloned_args = [
            arg.detach().clone().requires_grad_(arg.requires_grad) for arg in args
        ]
        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        opt_fn = torch.compile(fn, fullgraph=fullgraph, backend=backend)
        torch.manual_seed(0)
        result = opt_fn(*cloned_args)
        result.sum().backward()

        if not skip_check:
            self.assertEqual(result, expected)
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad)

    @requires_xpu_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout(self):
        """
        Regression: ``CUDARngStateHelper.get_torch_state_as_tuple`` must not
        raise ``RuntimeError('CUDA not available')`` when only XPU is present.

        Verifies that ``functionalize_rng_ops=True`` works end-to-end with
        dropout inside ``torch.utils.checkpoint.checkpoint`` on XPU.
        """

        def gn(x, y):
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device="xpu", requires_grad=True)
        y = torch.randn(4, 4, device="xpu", requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default
        )
        # philox_rand from fwd is reused in bwd; bwd must NOT call it again
        bw_compiler = functools.partial(
            count_ops, freq=0, op=torch.ops.rngprims.philox_rand.default
        )
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # skip_check=True: dropout decomp is known to diverge with eager
        self._validate(fn, backend, x, y, skip_check=True)

    @requires_xpu_and_triton
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function(self):
        """
        Verifies that ``functionalize_rng_ops=True`` works for a deterministic
        function (matmul + sigmoid) inside an activation checkpoint on XPU.

        Upstream:
          test/dynamo/test_higher_order_ops.py::ActivationCheckpointingTests::test_function
        """

        def gn(x, y):
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, requires_grad=True, device="xpu")
        y = torch.randn(4, 4, requires_grad=True, device="xpu")

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)


if __name__ == "__main__":
    if HAS_XPU_AND_TRITON:
        run_tests()
