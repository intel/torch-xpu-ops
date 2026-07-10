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


RE_NOT_IMPLEMENTED_MSG = re.compile(r"Could not run '([^']+)' with arguments ")

meta_function_expected_failures = {
    torch.Tensor.to_sparse: {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.allclose: {f64, f16, c128, c64, bf16, f32},
    torch.argwhere: {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.combinations: {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.corrcoef: {f64, i32, c128, i64, i16, u8, c64, bf16, f16, i8, f32},
    torch.cov: {f64, i32, c128, i64, i16, u8, c64, bf16, i8, f32, f16},
    torch.functional.istft: {f64, c64, c128, f32},
    torch.geqrf: {f64, c64, c128, f32},
    torch.masked_select: {f64, i32, c128, i64, i16, f16, u8, c64, bf16, b8, i8, f32},
    torch.nonzero: {f64, i32, c128, i64, i16, c32, f16, u8, c64, bf16, b8, i8, f32},
    torch.Tensor.nonzero: {
        f64,
        i32,
        c128,
        i64,
        i16,
        c32,
        f16,
        u8,
        c64,
        bf16,
        b8,
        i8,
        f32,
    },
    torch.Tensor.item: {f64, i32, c128, i64, i16, f16, u8, c32, c64, bf16, b8, i8, f32},
    torch.bincount: {i32, i64, u8, i16, i8},
    torch.functional.unique: {
        f64,
        i32,
        i64,
        u8,
        i16,
        f16,
        bf16,
        b8,
        i8,
        f32,
        u16,
        u32,
        u64,
    },
    torch.functional.unique_consecutive: {
        f64,
        i32,
        i64,
        u8,
        i16,
        f16,
        bf16,
        b8,
        i8,
        f32,
        u16,
        u32,
        u64,
    },
    torch.histogram: {f64, f32},
    torch.histogramdd: {f64, f32},
    torch.nn.functional.ctc_loss: {f64, f32},
    torch.nn.functional.gaussian_nll_loss: {f16, f64, bf16, f32},
    torch.linalg.lstsq: {f64, f32, c128, c64},
}

meta_function_expected_failures_conditional = {
    torch.repeat_interleave: lambda dtype, *args, **kwargs: (
        not isinstance(kwargs.get("repeats", None), int)
        and (kwargs.get("output_size", None) is None)
    ),
}

"""
# This is some sample code for how we could dump these dicts into YAML
# file for easier reading/writing
import yaml
print(yaml.dump(
  {resolve_name(k): [dtype_abbrs[d] for d in v]
   for k, v in meta_function_expected_failures.items()}, default_flow_style=None))
import sys
sys.exit()
"""

meta_function_skips = {
    torch.Tensor.__rmatmul__: {bf16, c128, f64, f32, f16, c64},
    torch.Tensor.matmul: {f64, f32, c128, c64},
    torch.functional.atleast_2d: {
        bf16,
        i8,
        c32,
        i64,
        u8,
        c128,
        b8,
        f64,
        i16,
        i32,
        f32,
        f16,
        c64,
    },
    torch.functional.atleast_3d: {
        bf16,
        i8,
        c32,
        i64,
        u8,
        c128,
        b8,
        f64,
        i16,
        i32,
        f32,
        f16,
        c64,
    },
    torch.functional.cartesian_prod: {
        bf16,
        i8,
        i64,
        u8,
        c128,
        b8,
        f64,
        i16,
        i32,
        f32,
        f16,
        c64,
    },
    torch.functional.einsum: {bf16, c128, f64, f32, f16, c64},
    torch.inner: {f16, bf16, i8, i64, u8, c128, f64, i16, f32, i32, c64},
    torch.linalg.matrix_norm: {c128, f32, c64, f64},
    torch.linalg.matrix_rank: {c128, c64},
    torch.linalg.svd: {c128, c64},
    torch.matmul: {bf16, c128, f64, f32, f16, c64},
    torch.nanquantile: {f64, f32},
    torch.narrow: {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c32, c64},
    torch.nn.functional.batch_norm: {f64, f32},
    torch.nn.functional.binary_cross_entropy: {bf16, f64, f32, f16},
    torch.nn.functional.dropout3d: {bf16, f64, f32, f16},
    torch.nn.functional.local_response_norm: {bf16, f64, f32, f16},
    torch.svd: {c128, c64},
    torch.take_along_dim: {bf16, i8, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.vstack: {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.diff: {b8},
    torch.equal: {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.nanmean: {bf16, f64, f32, f16, c32, c64, c128},
    torch.nn.functional.cross_entropy: {bf16, f64, f32},
    torch.nn.functional.nll_loss: {bf16, f64, f32},
    torch.linalg.cond: {c128, c64, f32, f64},
    torch.linalg.vecdot: {bf16, f64, f32, f16},
    torch.empty: {bf16, i8, c32, i64, u8, c128, b8, f64, i16, i32, f32, f16, c64},
    torch.Tensor.addbmm_: {bf16, c128, c64, f32, f64, i16, i32, i64, i8, u8},
    torch.nn.functional.one_hot: {i64},
}


meta_function_device_expected_failures = defaultdict(dict)
meta_function_device_expected_failures_only_outplace = defaultdict(dict)
meta_function_device_skips = defaultdict(dict)

meta_function_device_expected_failures["cpu"] = {
    # TODO: The decomps for these batch norm ops return different dtypes depending
    # on the device. We should make this work better with meta tensors.
    torch.native_batch_norm: {bf16, f16},
    torch._native_batch_norm_legit: {bf16, f16},
    torch.ops.aten._batch_norm_with_update: {bf16, f16},
    torch.native_layer_norm: {bf16, f16},
}

meta_function_device_expected_failures["cuda"] = {
    torch.corrcoef: {bf16, f16},  # aten::_local_scalar_dense
    torch.cov: {f16},  # aten::_local_scalar_dense
    torch.functional.unique: {f16},  # aten::_unique2, aten::unique_dim
    torch.functional.unique_consecutive: {f16},  # aten::unique_consecutive
    torch.geqrf: {f32, f64},  # aten::geqrf
}

meta_function_device_skips["cpu"] = {
    # TODO: The decomps for these batch norm ops return different dtypes depending
    # on the device. We should make this work better with meta tensors.
    torch.native_batch_norm: {f32, f64},
    torch._native_batch_norm_legit: {f32, f64},
    torch.ops.aten._batch_norm_with_update: {f32, f64},
}

meta_function_device_skips["cuda"] = {
    torch.inner: {f16},
    torch.linalg.matrix_rank: {f32, f64},
    torch.linalg.svd: {f32, f64},
    torch.nn.functional.cross_entropy: {f16},
    torch.nn.functional.interpolate: {f16},
    torch.nn.functional.nll_loss: {f16},
    torch.svd: {f32, f64},
}

meta_function_device_expected_failures["xpu"] = meta_function_device_expected_failures[
    "cuda"
]
meta_function_device_skips["xpu"] = meta_function_device_skips["cuda"]


# This is a __torch_function__ mode that, when enabled, interposes every
# Torch API call and runs the operator as normal, and then reruns it
# with meta inputs, and then checks that everything about the output agrees.
# Most of the logic deals with faithfully replicating the original tensor
# as a meta tensor, which is nontrivial because there are a lot of subsystems
# that may potentially be exercised.
#
# That being said, this class is a little overkill for what it is doing in
# this test file (since I could have just inlined __torch_function__ on the
# OpInfo call, and OpInfos generally have very regular inputs), but it will be
# useful for more comprehensive testing e.g., as seen in
# https://github.com/pytorch/pytorch/pull/75994  The big benefit is it is
# A LOT more efficient that torch dispatch mode (at the cost of less coverage)
class MetaCrossRefFunctionMode(torch.overrides.TorchFunctionMode):
    test_case: TestCase
    device_type: str
    dtype: torch.dtype

    def __init__(self, test_case, *, device, dtype, inplace):
        self.test_case = test_case
        self.device_type = torch.device(device).type
        self.dtype = dtype
        self.inplace = inplace

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        if (
            torch.jit.is_tracing()
            or isinstance(func, torch.ScriptMethod)
            or
            # meta converter doesn't work correctly when no_dispatch() is on, so
            # skip running the crossref test in this case
            torch._C._dispatch_tls_local_exclude_set().has(torch._C.DispatchKey.Python)
        ):
            return func(*args, **kwargs)

        if self.dtype in meta_function_skips.get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_function_device_skips[self.device_type].get(
            func, set()
        ):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_function_expected_failures.get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif self.dtype in meta_function_device_expected_failures[self.device_type].get(
            func, set()
        ):
            test_expect = TestExpect.XFAILURE
        elif meta_function_expected_failures_conditional.get(
            func, lambda *_, **__: False
        )(self.dtype, *args, **kwargs):
            test_expect = TestExpect.XFAILURE
        elif (
            not self.inplace
            and self.dtype
            in meta_function_device_expected_failures_only_outplace[
                self.device_type
            ].get(func, set())
        ):
            test_expect = TestExpect.XFAILURE
        else:
            test_expect = TestExpect.SUCCESS

        return run_meta_crossref(
            self.test_case,
            test_expect,
            func,
            args,
            kwargs,
            dtype=self.dtype,
            device_type=self.device_type,
            run_symbolic_meta=False,
        )


# these always fail
meta_dispatch_expected_failures = {
    aten.allclose.default: {
        f16,
        bf16,
        f32,
        f64,
        c64,
        c128,
    },  # NotImplementedError: 'aten::_local_scalar_dense'
    aten.geqrf.default: {c64, c128, f64, f32},
    aten.linalg_lstsq.default: {c64, c128, f64, f32},
    aten.masked_select.default: {
        c64,
        f16,
        i8,
        f64,
        c128,
        i64,
        bf16,
        f32,
        i32,
        b8,
        i16,
        u8,
    },
    aten.masked_select.out: {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.nonzero.default: {
        c64,
        f16,
        i8,
        f64,
        c128,
        i64,
        bf16,
        f32,
        i32,
        c32,
        b8,
        i16,
        u8,
    },
    aten.nonzero.out: {c64, f16, i8, f64, c128, i64, bf16, f32, i32, c32, b8, i16, u8},
    aten._to_sparse.default: {
        c64,
        f16,
        i8,
        f64,
        c128,
        i64,
        bf16,
        f32,
        i32,
        b8,
        i16,
        u8,
    },
    aten._to_sparse.sparse_dim: {
        c64,
        f16,
        i8,
        f64,
        c128,
        i64,
        bf16,
        f32,
        i32,
        b8,
        i16,
        u8,
    },
    aten._ctc_loss.Tensor: {f32, f64},  # Shape of second output depends on data.
    aten._histogramdd_bin_edges.default: {f32, f64},
    aten._histogramdd_from_bin_cts.default: {f32, f64},
    aten._histogramdd_from_bin_tensors.default: {f32, f64},
    aten._local_scalar_dense.default: {
        c32,
        c64,
        f16,
        i8,
        f64,
        c128,
        i64,
        bf16,
        f32,
        i32,
        b8,
        i16,
        u8,
    },
    aten._unique2.default: {
        i8,
        f64,
        i64,
        f16,
        bf16,
        f32,
        i32,
        b8,
        i16,
        u8,
        u16,
        u32,
        u64,
    },
    aten.bincount.default: {i64, i8, i32, i16, u8},
    aten.equal.default: {c64, f16, i8, f64, c128, i64, bf16, f32, i32, b8, i16, u8},
    aten.histogram.bin_ct: {f32, f64},
    aten.histogram.bins_tensor: {f32, f64},
    aten.unique_consecutive.default: {
        i8,
        f64,
        i64,
        f16,
        bf16,
        f32,
        i32,
        b8,
        i16,
        u8,
        u16,
        u32,
        u64,
    },
    aten.unique_dim.default: {
        i8,
        f64,
        i64,
        f16,
        bf16,
        f32,
        i32,
        b8,
        i16,
        u8,
        u16,
        u32,
        u64,
    },
    aten.upsample_nearest3d.vec: {bf16, f32, f64, u8},
}

# these sometimes pass and sometimes fail
meta_dispatch_skips = {
    aten.index.Tensor: {
        i64,
        bf16,
        f16,
        u8,
        b8,
        f32,
        i8,
        f64,
        i16,
        i32,
        c32,
        c64,
        c128,
    },  # at::nonzero doesn't have a Meta function
    aten._to_copy.default: {
        i64,
        bf16,
        f16,
        u8,
        b8,
        f32,
        i8,
        f64,
        i16,
        i32,
        c32,
        c64,
        c128,
    },
    aten.empty.memory_format: {
        b8,
        bf16,
        c128,
        c64,
        c32,
        f16,
        f32,
        f64,
        i16,
        i32,
        i64,
        i8,
        u8,
    },
    aten.addbmm_.default: {bf16, c128, c64, f32, f64, i16, i32, i64, i8, u8},
}

# For CompositeImplicitAutograd functions that fail before hitting the Mode
meta_dispatch_early_skips = set(
    {
        torch.Tensor.float_power_,
        # Errors out in one of the tests, while ProxyTensor passes...
        torch.Tensor.cumprod_,
        torch.Tensor.cumsum_,
    }
)

meta_inplace_skips = set(
    {
        # Errors out in one of the tests, while ProxyTensor passes...
        torch.Tensor.cumprod_,
        torch.Tensor.cumsum_,
    }
)

meta_dispatch_device_expected_failures = defaultdict(dict)
meta_dispatch_device_skips = defaultdict(dict)

meta_dispatch_device_expected_failures["cpu"] = {
    # TODO: The decomps for these batch norm ops return different dtypes depending
    # on the device. We should make this work better with meta tensors.
    aten.native_batch_norm.default: {bf16, f16},
    aten._native_batch_norm_legit.default: {bf16, f16},
    aten._native_batch_norm_legit.no_stats: {bf16, f16},
    aten._batch_norm_with_update.default: {bf16, f16},
    aten.native_layer_norm.default: {bf16, f16},
}

meta_dispatch_device_expected_failures["cuda"] = {
    aten._unique2.default: {f16},  # aten::_unique2
    aten._use_cudnn_ctc_loss.default: {f32, f64},  # aten::_use_cudnn_ctc_loss
    aten._use_cudnn_ctc_loss.Tensor: {f32, f64},  # aten::_use_cudnn_ctc_loss.Tensor
    aten.cudnn_grid_sampler.default: {f16, f32, f64},  # aten::cudnn_grid_sampler
    aten.geqrf.default: {f32, f64},  # aten::geqrf
    aten.linalg_eigvalsh.out: {f32, f64},  # aten::linalg_eigvalsh.out
    aten.log_sigmoid_forward.default: {bf16, f16, f64, f32},
    aten.log_sigmoid_forward.output: {
        bf16,
        f16,
        f64,
        f32,
    },  # aten::log_sigmoid_forward.output
    aten.unique_consecutive.default: {f16},  # aten::unique_consecutive
    aten.unique_dim.default: {f16},  # aten::unique_dim
    aten.upsample_nearest3d.vec: {f16},  # aten::upsample_nearest3d.vec
}

meta_dispatch_device_skips["cpu"] = {
    aten._embedding_bag_forward_only.default: {bf16, f16, f32, f64},
    # TODO: The decomps for these batch norm ops return different dtypes depending
    # on the device. We should make this work better with meta tensors.
    aten.native_batch_norm.default: {f32, f64},
    aten._native_batch_norm_legit.default: {f32, f64},
    aten._native_batch_norm_legit.no_stats: {f32, f64},
    aten._batch_norm_with_update.default: {f32, f64},
    # If the computation dtype is different from the input
    # dtype this will fail. CPU execution may also have a
    # a different output from other devices.
    aten.native_batch_norm.out: {bf16, f16, f32, f64},
}

meta_dispatch_device_skips["cuda"] = {
    aten._conj.default: {c32, f16},  # file issue
    aten._linalg_svd.default: {c64, c128},  # aten::linalg_eigvalsh.out
    aten.cudnn_batch_norm.default: {f32, f64},
    aten.log_softmax.int: {c32, c64},
    aten.softmax.int: {c32, c64},
    aten.softmax.int: {c32, c64},
    # ROCm stuff; technically this should be expected failure but it's
    # not worth it; these should get unified anyway
    aten.miopen_batch_norm.default: {f32},
}

meta_dispatch_device_skips["xpu"] = meta_dispatch_device_skips["cuda"]
meta_dispatch_device_expected_failures["xpu"] = meta_dispatch_device_expected_failures[
    "cuda"
]


def get_strided_args(args):
    def get_strided_variants(t, include_storage_offset=False):
        variants = []

        # contiguous
        variants.append(t)

        # transposed
        if t.ndim > 1:
            perm = list(reversed(range(t.ndim)))
            transposed = (
                torch.empty(
                    t.shape[::-1],
                    device=t.device,
                    dtype=t.dtype,
                    requires_grad=t.requires_grad,
                )
                .permute(perm)
                .copy_(t)
            )
            variants.append(transposed)

        # nondense
        if t.ndim > 0:
            nondense = torch.repeat_interleave(t, 2, dim=-1)[..., ::2]
            variants.append(nondense)

        # channel_last
        if t.ndim == 4:
            variants.append(t.contiguous(memory_format=torch.channels_last))

        # channel_last_3d
        if t.ndim == 5:
            variants.append(t.contiguous(memory_format=torch.channels_last_3d))

        # storage_offset
        if include_storage_offset:
            buffer = torch.empty(
                t.numel() + 1,
                device=t.device,
                dtype=t.dtype,
                requires_grad=t.requires_grad,
            )
            buffer = buffer.as_strided(t.shape, t.stride(), storage_offset=1)
            buffer.copy_(t)
            variants.append(buffer)

        return variants

    strided_args = []
    for arg in args:
        if (
            isinstance(arg, torch.Tensor)
            and not arg.is_sparse_csr
            and arg.is_contiguous()
        ):
            strided_arg_variants = get_strided_variants(arg)
        else:
            strided_arg_variants = [arg]
        strided_args.append(strided_arg_variants)

    yield from itertools.product(*strided_args)


class MetaCrossRefDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    test_case: TestCase
    device: torch.device
    dtype: torch.dtype
    aten_olp_no_out_overload: set = set()

    def __init__(
        self,
        test_case,
        *,
        device,
        dtype,
        symbolic_meta: bool,
        inplace: bool,
        supports_out: bool,
    ):
        self.test_case = test_case
        # save TLS
        self.precision = test_case.precision
        self.rel_tol = test_case.rel_tol
        self.device_type = torch.device(device).type
        self.dtype = dtype
        self.symbolic_meta = symbolic_meta
        self.inplace = inplace
        self.supports_out = supports_out

    @staticmethod
    def try_resolve_aten_out_overload(ol, args, kwargs, num_outputs):
        ol_args = ol._schema.arguments
        olp: OpOverloadPacket = ol._overloadpacket

        if olp in MetaCrossRefDispatchMode.aten_olp_no_out_overload:
            return (None, None, None)

        candidate_ols = []
        for candidate_ol_name in olp.overloads():
            candidate_ol = getattr(olp, candidate_ol_name)
            if any(arg.is_out for arg in candidate_ol._schema.arguments):
                candidate_ols.append(candidate_ol)

        if not candidate_ols:
            MetaCrossRefDispatchMode.aten_olp_no_out_overload.add(olp)
            return (None, None, None)

        # Now match based on args, kwargs and number of required outputs
        candidate_ol: OpOverload = None
        for candidate_ol in candidate_ols:
            candidate_ol_args = candidate_ol._schema.arguments

            if len(args) >= len(candidate_ol_args):
                continue

            # Positional arguments must have the same type
            if not all(
                ol_args[pos_arg_ind].type == candidate_ol_args[pos_arg_ind].type
                for pos_arg_ind in range(len(args))
            ):
                continue

            # Number of outputs must match
            candidate_out_names = [
                out_arg.name
                for out_arg in candidate_ol_args[-num_outputs:]
                if out_arg.is_out
            ]
            if len(candidate_out_names) != num_outputs:
                continue

            # Now try and match kwargs. Just need to ensure that the
            # remaining kwargs allow an out overload to be called. For example
            # we can throw away parameters like `dtype` that may be passed to the
            # functional version of the op since the `dtype` will already be present
            # in the `out` argument
            new_kwargs = {}
            kwargs_match = True
            for arg in candidate_ol_args[len(args) : -num_outputs]:
                if arg.name not in kwargs:
                    if arg.has_default_value():
                        new_kwargs[arg.name] = arg.default_value
                    elif isinstance(arg.type, torch.OptionalType):
                        if isinstance(arg.type.getElementType(), torch.BoolType):
                            new_kwargs[arg.name] = False
                        else:
                            new_kwargs[arg.name] = None
                    else:
                        kwargs_match = False
                        break
                else:
                    new_kwargs[arg.name] = kwargs[arg.name]

            if kwargs_match:
                return candidate_ol, candidate_out_names, new_kwargs

        return None, None, None

    def _get_expected_test_result(self, func: OpOverload):
        if self.dtype in meta_dispatch_skips.get(func, set()):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_dispatch_device_skips[self.device_type].get(
            func, set()
        ):
            test_expect = TestExpect.SKIP
        elif self.dtype in meta_dispatch_expected_failures.get(func, set()):
            test_expect = TestExpect.XFAILURE
        elif self.dtype in meta_dispatch_device_expected_failures[self.device_type].get(
            func, set()
        ):
            test_expect = TestExpect.XFAILURE
        else:
            test_expect = TestExpect.SUCCESS
        return test_expect

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self.test_case.precision = self.precision
        self.test_case.rel_tol = self.rel_tol

        test_expect = self._get_expected_test_result(func)

        expected = run_meta_crossref(
            self.test_case,
            test_expect,
            func,
            args,
            kwargs,
            dtype=self.dtype,
            device_type=self.device_type,
            run_symbolic_meta=self.symbolic_meta,
        )

        # This is to test torch ops that do not have an out parameter but have
        # aten op overloads that have out parameters. Additionally, Python decompositions
        # may register OpOverloadPacket's so decompositions need to be tested
        # to ensure all OpOverloads still function for the Meta key (e.g. if a python decomposition
        # is registered for an aten op aten.foo with overloads [default, out], the python
        # function needs to support receiving `out` arguments)
        if (
            not self.inplace
            and not self.supports_out
            and test_expect == TestExpect.SUCCESS
            and (torch.is_tensor(expected) or isinstance(expected, Iterable))
        ):
            # check to see if there is a potential out overload
            num_outputs = 1 if torch.is_tensor(expected) else len(expected)
            (
                func_out_overload,
                out_param_names,
                kwargs,
            ) = self.try_resolve_aten_out_overload(func, args, kwargs, num_outputs)

            if func_out_overload:
                if num_outputs == 1:
                    kwargs[out_param_names[0]] = expected
                else:
                    for ind, out_param_name in enumerate(out_param_names):
                        kwargs[out_param_name] = expected[ind]

                test_expect = self._get_expected_test_result(func_out_overload)

                run_meta_crossref(
                    self.test_case,
                    test_expect,
                    func_out_overload,
                    args,
                    kwargs,
                    dtype=self.dtype,
                    device_type=self.device_type,
                    run_symbolic_meta=self.symbolic_meta,
                )

        return expected


# NB: we're running these tests only on CUDA because there are some
# inconsistencies between CUDA and CPU, and running on CUDA makes it easier
# to ignore the CPU case when inconsistencies arise.  Ideally we deal
# with the inconsistencies but this takes time.
@unMarkDynamoStrictTest
class TestMeta(TestCase):
    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            if isinstance(t, list):
                return inplace_variant([x.clone() for x in t], *args, **kwargs)
            else:
                return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    @skipIfCrossRef
    @suppress_warnings
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_meta_outplace(self, device, dtype, op):
        if "_scaled_mm" in op.name:
            raise unittest.SkipTest("_scaled_mm dose not support meta device")
        skip_op_names = (
            "fft.ihfft",
            "fft.ihfft2",
            "linalg.lu_solve",
        )
        if TEST_WITH_TORCHDYNAMO and op.name in skip_op_names:
            raise unittest.SkipTest("flaky")
        # run the OpInfo sample inputs, cross-referencing them with the
        # meta implementation and check the results are the same.  All
        # the heavy lifting happens in MetaCrossRefFunctionMode
        func = op.get_op()
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            with MetaCrossRefFunctionMode(
                self, dtype=dtype, device=device, inplace=False
            ):
                expected = func(*args, **kwargs)
                if isinstance(expected, torch.Tensor) and op.supports_out:
                    func(*args, **kwargs, out=expected)

            # Special test for functions taking "device" kwarg
            # The crossref tests that replacing the device with "meta" works
            # This part makes sure that *_like functions work well with a "meta"
            # Tensor and their original device argument.
            if "device" in kwargs and "_like" in op.name:
                with torch.random.fork_rng():
                    torch.manual_seed(123)
                    ref = func(*args, **kwargs)

                # *_like functions take a Tensor as first argument
                assert isinstance(args[0], torch.Tensor)
                with torch.random.fork_rng():
                    torch.manual_seed(123)
                    args[0] = args[0].to(device="meta")
                    meta = func(*args, **kwargs)

                # empty_like is not deterministic
                if op.name != "empty_like":
                    self.assertEqual(ref, meta)

    @skipIfCrossRef
    @suppress_warnings
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_meta_inplace(self, device, dtype, op):
        func = op.get_inplace()
        if not func:
            self.skipTest("No inplace variable for this op")
        if op.promotes_int_to_float and not dtype.is_floating_point:
            self.skipTest(
                "Op promotes to float, which is impossible for inplace with non-float input"
            )
        if func in meta_inplace_skips:
            self.skipTest("Skipped")
        func = self._get_safe_inplace(func)
        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            if sample_input.broadcasts_input:
                continue
            args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs
            with MetaCrossRefFunctionMode(
                self, dtype=dtype, device=device, inplace=True
            ):
                expected = func(*args, **kwargs)

    def _run_dispatch_meta_test(
        self, device, dtype, op, symbolic_meta, inplace, all_stride_variants=False
    ):
        if "_scaled_mm" in op.name:
            raise unittest.SkipTest("_scaled_mm dose not support meta device")
        if inplace:
            func = op.get_inplace()
            if not func:
                self.skipTest("No inplace variable for this op")
            if op.promotes_int_to_float and not dtype.is_floating_point:
                self.skipTest(
                    "Op promotes to float, which is impossible for inplace with non-float input"
                )
        else:
            func = op.get_op()

        if func in meta_dispatch_early_skips:
            self.skipTest("Function is in dispatch early skips")

        if inplace:
            func = self._get_safe_inplace(func)

        samples = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in samples:
            if inplace and sample_input.broadcasts_input:
                continue

            sample_args = [sample_input.input] + list(sample_input.args)
            kwargs = sample_input.kwargs

            if (
                all_stride_variants
                and sum(isinstance(arg, torch.Tensor) for arg in sample_args) <= 5
            ):
                # test inputs <= 5 tensors to avoid combinatorial explosion
                strided_args = get_strided_args(sample_args)
            else:
                strided_args = [sample_args]

            for args in strided_args:
                with MetaCrossRefDispatchMode.push(
                    self,
                    dtype=dtype,
                    device=device,
                    symbolic_meta=symbolic_meta,
                    inplace=inplace,
                    supports_out=op.supports_out,
                ):
                    expected = func(*args, **kwargs)

                    if (
                        not inplace
                        and isinstance(expected, torch.Tensor)
                        and op.supports_out
                    ):
                        func(*args, **kwargs, out=expected)

    @skipIfCrossRef
    @suppress_warnings
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_dispatch_meta_outplace(self, device, dtype, op):
        self._run_dispatch_meta_test(
            device, dtype, op, symbolic_meta=False, inplace=False
        )

    @skipIfCrossRef
    @suppress_warnings
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_dispatch_meta_inplace(self, device, dtype, op):
        self._run_dispatch_meta_test(
            device, dtype, op, symbolic_meta=False, inplace=True
        )

    @skipIfCrossRef
    @suppress_warnings
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_dispatch_symbolic_meta_outplace(self, device, dtype, op):
        self._run_dispatch_meta_test(
            device, dtype, op, symbolic_meta=True, inplace=False
        )

    @skipIfCrossRef
    @suppress_warnings
    @ops(itertools.chain(op_db, foreach_op_db))
    def test_dispatch_symbolic_meta_inplace(self, device, dtype, op):
        self._run_dispatch_meta_test(
            device, dtype, op, symbolic_meta=True, inplace=True
        )

    @skipIfCrossRef
    @suppress_warnings
    # only test one dtype, as output stride behavior is the same for all dtypes
    @ops(itertools.chain(op_db, foreach_op_db), dtypes=OpDTypes.any_common_cpu_cuda_one)
    # Only test on CUDA, as CUDA kernel's stride is the reference
    @onlyOn(["cuda", "xpu"])
    def test_dispatch_symbolic_meta_outplace_all_strides(self, device, dtype, op):
        self._run_dispatch_meta_test(
            device,
            dtype,
            op,
            symbolic_meta=True,
            inplace=False,
            all_stride_variants=True,
        )

    @skipIfCrossRef
    @suppress_warnings
    # only test one dtype, as output stride behavior is the same for all dtypes
    @ops(itertools.chain(op_db, foreach_op_db), dtypes=OpDTypes.any_common_cpu_cuda_one)
    # Only test on CUDA, as CUDA kernel's stride is the reference
    @onlyOn(["cuda", "xpu"])
    def test_dispatch_symbolic_meta_inplace_all_strides(self, device, dtype, op):
        self._run_dispatch_meta_test(
            device,
            dtype,
            op,
            symbolic_meta=True,
            inplace=True,
            all_stride_variants=True,
        )

    @skipIfCrossRef
    @suppress_warnings
    # only test one dtype, as output stride behavior is the same for all dtypes
    @ops(binary_ufuncs, allowed_dtypes=(torch.float32,))
    # Only test on CUDA, as CUDA kernel's stride is the reference
    @onlyOn(["cuda", "xpu"])
    def test_binary_ufuncs_mixed_dtype(self, device, dtype, op):
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

        self._run_dispatch_meta_test(
            device, dtype, op, symbolic_meta=True, inplace=False
        )

    def test_empty_quantized(self):
        r = torch.empty(2**52, device="meta", dtype=torch.qint8)
        self.assertEqual(r.device.type, "meta")

    def test_nan_to_num(self):
        t = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 3.14], device="meta"
        )
        r = t.nan_to_num()
        self.assertEqual(r.device.type, "meta")

    def test_inplace_masked_fill_error(self):
        t = torch.randn(3, 3, device="meta")
        with self.assertRaisesRegex(RuntimeError, "doesn't match the broadcast"):
            t.masked_fill_((t > 0).unsqueeze(0), 0.1)

    def test_inplace_bin_ops_error(self):
        t = torch.randn(3, 3, device="meta")
        for op in (
            torch.Tensor.add_,
            torch.Tensor.sub_,
            torch.Tensor.mul_,
            torch.Tensor.div_,
            torch.Tensor.logical_and_,
            torch.Tensor.logical_or_,
            torch.Tensor.logical_xor_,
        ):
            with self.assertRaisesRegex(RuntimeError, "doesn't match the broadcast"):
                op(t, t.clone().unsqueeze(0))

    @onlyCPU
    def test_meta_autograd_no_error(self):
        with torch.library._scoped_library("meta_test", "DEF") as lib:
            with torch.library._scoped_library("meta_test", "IMPL", "CPU") as impl_cpu:
                with torch.library._scoped_library(
                    "meta_test", "IMPL", "Meta"
                ) as impl_meta:

                    def foo_impl(x):
                        return x + 1

                    lib.define("foo(Tensor a) -> Tensor")
                    impl_meta.impl("foo", foo_impl)
                    impl_cpu.impl("foo", foo_impl)

                    a = torch.ones(2, device="meta")
                    # The point of the test is that this should not error:
                    # We have a fallthrough kernel registered to the AutogradMeta
                    # key for custom ops, so it's fine that `foo()` doesn't have
                    # an autograd kernel.
                    b = torch.ops.meta_test.foo.default(a)

    def test_huber_loss_backward(self):
        inps = [torch.rand(2**52, device="meta") for _ in range(3)]
        r = torch.ops.aten.huber_loss_backward(*inps, 0, 1.0)
        self.assertEqual(r.device.type, "meta")
        self.assertEqual(r.shape, inps[0].shape)

    def _norm_backwards_test_helper(self, op, args, output_mask, expected_shapes):
        dtype = torch.float32
        device = "meta"

        # test functional call
        grads = op(*args, output_mask)

        def assertEqualShapes(res, exp):
            self.assertIsNone(res) if exp is None else self.assertEqual(exp, res.shape)

        assertEqualShapes(grads[0], expected_shapes[0])
        assertEqualShapes(grads[1], expected_shapes[1])
        assertEqualShapes(grads[2], expected_shapes[2])

        out_kwargs = {
            f"out{i}": torch.empty(0, device=device, dtype=dtype)
            for i in range(len(output_mask))
        }

        # test call with out parameters
        grads = op(*args, output_mask, **out_kwargs)

        def assertEqualShapes(res, exp):
            self.assertEqual(exp, res.shape) if exp is not None else True

        assertEqualShapes(out_kwargs["out0"], expected_shapes[0])
        assertEqualShapes(out_kwargs["out1"], expected_shapes[1])
        assertEqualShapes(out_kwargs["out2"], expected_shapes[2])

    @onlyCPU
    @parametrize(
        "output_mask",
        list(itertools.product([True, False], [True, False], [True, False])),
    )
    def test_layer_norm_backward(self, output_mask):
        from torch.testing._internal.common_methods_invocations import (
            sample_inputs_layer_norm,
        )

        device = "meta"
        dtype = torch.float32

        samples = sample_inputs_layer_norm(None, device, dtype, requires_grad=False)

        for sample in samples:
            with self.subTest(sample=sample):
                # handle optional weight and bias
                if len(sample.args) != 3:
                    sample.args = (*sample.args, *([None] * (3 - len(sample.args))))

                grad_out = torch.ones_like(sample.input)
                normalized_shape, weight, bias = sample.args
                ndims_after_reduction = sample.input.ndim - len(normalized_shape)
                mean_shape = grad_out.shape[:ndims_after_reduction]
                mean = torch.zeros(mean_shape, device=device, dtype=dtype)
                rstd = torch.zeros(mean_shape, device=device, dtype=dtype)

                expected_shapes = (
                    sample.input.shape if output_mask[0] else None,
                    weight.shape if output_mask[1] and weight is not None else None,
                    bias.shape if output_mask[2] and bias is not None else None,
                )

                args = [
                    grad_out,
                    sample.input,
                    normalized_shape,
                    mean,
                    rstd,
                    weight,
                    bias,
                ]

                self._norm_backwards_test_helper(
                    torch.ops.aten.native_layer_norm_backward,
                    args,
                    output_mask,
                    expected_shapes,
                )

    @onlyCPU
    @parametrize(
        "output_mask",
        list(itertools.product([True, False], [True, False], [True, False])),
    )
    def test_group_norm_backward(self, output_mask):
        from torch.testing._internal.common_methods_invocations import (
            sample_inputs_group_norm,
        )

        # input, (args) num_groups, (kwargs) weight, bias eps
        device = "meta"
        dtype = torch.float32
        samples = sample_inputs_group_norm(None, device, dtype, requires_grad=False)

        for sample in samples:
            with self.subTest(sample=sample):
                grad_out = torch.ones_like(sample.input)
                N, C = sample.input.shape[:2]
                HxW = torch.prod(
                    torch.as_tensor(sample.input.shape[2:]), dtype=torch.int32
                ).item()
                group = sample.args[0]
                mean = torch.zeros((N, group), device=device, dtype=dtype)
                rstd = torch.zeros((N, group), device=device, dtype=dtype)
                weight = torch.zeros((C), device=device, dtype=dtype)

                args = [grad_out, sample.input, mean, rstd, weight, N, C, HxW, group]

                expected_shapes = (
                    sample.input.shape if output_mask[0] else None,
                    weight.shape if output_mask[1] else None,
                    weight.shape if output_mask[2] else None,
                )

                # test functional call
                self._norm_backwards_test_helper(
                    torch.ops.aten.native_group_norm_backward,
                    args,
                    output_mask,
                    expected_shapes,
                )

    @onlyCPU
    @parametrize(
        "output_mask", list(itertools.product([True], [True, False], [True, False]))
    )
    def test_batch_norm_backward(self, output_mask):
        from torch.testing._internal.common_methods_invocations import (
            sample_inputs_batch_norm,
        )

        # input, (args) num_groups, (kwargs) weight, bias eps
        device = "meta"
        dtype = torch.float32
        samples = sample_inputs_batch_norm(None, device, dtype, requires_grad=False)

        for sample in samples:
            with self.subTest(sample=sample):
                if sample.input.dim() < 2:
                    continue

                grad_out = torch.ones_like(sample.input)
                running_mean, running_var, weight, bias = sample.args
                train = sample.kwargs.get("training", True)
                save_mean = (
                    torch.zeros((sample.input.shape[1],), device=device, dtype=dtype)
                    if train
                    else None
                )
                save_invstd = (
                    torch.zeros((sample.input.shape[1],), device=device, dtype=dtype)
                    if train
                    else None
                )

                args = [
                    grad_out,
                    sample.input,
                    weight,
                    running_mean,
                    running_var,
                    save_mean,
                    save_invstd,
                    train,
                    sample.kwargs.get("eps", 1e-5),
                ]

                expected_shapes = (
                    sample.input.shape,
                    torch.Size([sample.input.shape[1]]) if output_mask[1] else None,
                    torch.Size([sample.input.shape[1]]) if output_mask[2] else None,
                )

                self._norm_backwards_test_helper(
                    torch.ops.aten.native_batch_norm_backward,
                    args,
                    output_mask,
                    expected_shapes,
                )

    def test_fill__alias_relationship(self):
        inps = torch.rand(2**52, device="meta")
        r = torch.ops.aten.fill_(inps, 1.0)
        # aten.fill_ returns an alias
        self.assertEqual(id(inps), id(r))

        # aten.fill returns a new tensor
        r2 = torch.ops.aten.fill(inps, 1.0)
        self.assertNotEqual(id(inps), id(r2))

    def test_meta__fused_moving_avg_obs_fq_helper(self, device):
        from torch.ao.quantization import FusedMovingAvgObsFakeQuantize

        to_meta = MetaConverter()

        x = torch.randn(5, 5, device=device)
        running_min_op = torch.tensor(float("inf"), device=device)
        running_max_op = torch.tensor(float("-inf"), device=device)
        avg_const = 0.01
        scale = torch.tensor([1.0], device=device)
        zero_point = torch.tensor([0], dtype=torch.int, device=device)

        mod = FusedMovingAvgObsFakeQuantize()
        torch.ao.quantization.enable_fake_quant(mod)
        torch.ao.quantization.enable_observer(mod)
        mod.to(device)

        meta_x = to_meta(x)

        args = [
            x,
            mod.observer_enabled,
            mod.fake_quant_enabled,
            running_min_op,
            running_max_op,
            scale,
            zero_point,
            avg_const,
            0,
            255,
            0,
        ]

        meta_args = args.copy()
        meta_args[0] = meta_x

        kwargss = [
            {},
            {"per_row_fake_quant": False, "symmetric_quant": False},
            {"per_row_fake_quant": False, "symmetric_quant": True},
        ]

        for kwargs in kwargss:
            ref_out = aten._fused_moving_avg_obs_fq_helper.default(*args, **kwargs)
            meta_out = aten._fused_moving_avg_obs_fq_helper.default(
                *meta_args, **kwargs
            )

            self.assertEqual(ref_out[0].size(), meta_out[0].size())
            self.assertEqual(ref_out[0].stride(), meta_out[0].stride())
            self.assertEqual(ref_out[1].size(), meta_out[1].size())
            self.assertEqual(ref_out[1].stride(), meta_out[1].stride())

    def test_cdist_forward(self, device):
        to_meta = MetaConverter()
        x1 = torch.rand([3, 2], device=device)
        x2 = torch.rand([2, 2], device=device)
        p = 2.0
        for compute_mode in (None, 1, 2):
            ref = aten._cdist_forward.default(x1, x2, p, compute_mode)
            res = aten._cdist_forward.default(to_meta(x1), to_meta(x2), p, compute_mode)
            self.assertEqual(res.device.type, "meta")
            self.assertEqual(ref.shape, res.shape)

    def test_quantized_embedding_bag(self):
        tab_shape = [8, 128]
        emb_size, ind_len, off_len = tab_shape[0], 32, 33
        f_table = torch.from_numpy(
            (np.random.random_sample(tab_shape) + 1).astype(np.float32)
        )
        q_table = torch.ops.quantized.embedding_bag_byte_prepack(f_table)
        indices = torch.from_numpy(
            np.random.randint(low=0, high=emb_size, size=ind_len)
        ).int()
        max_length = len(indices) // (off_len - 1)
        if max_length > 20:
            max_length = 20
        np_lengths = np.random.randint(0, max_length + 1, size=off_len - 1).astype(
            np.int32
        )
        offsets = torch.cat(
            [torch.zeros([1]), torch.cumsum(torch.from_numpy(np_lengths), 0)]
        ).int()

        eb = torch.ops.quantized.embedding_bag_byte_rowwise_offsets(
            q_table.to(device="meta"),
            indices.to(device="meta"),
            offsets.to(device="meta"),
            mode=0,  # sum
            per_sample_weights=None,
            include_last_offset=True,
        )
        self.assertEqual(eb.shape, [32, 128])
        self.assertEqual(eb.dtype, torch.float32)
        self.assertEqual(eb.untyped_storage().data_ptr(), 0)

    # Tests mean and max.
    # Can't easily test sum, because there is a fast path for sum which
    # causes offset2bag to not get allocated... but the backward function
    # needs it, and the offset2bag computation lives inside the
    # derivatives.yaml formula directly, so there is no way to access it.
    # To test sum, need to manually compute offset2bag
    @parametrize("mode", [1, 2])
    def test_embedding_bag_dense_backward(self, mode):
        weight = torch.randn(4, 3, requires_grad=True)
        indices = torch.tensor([1, 0, 2, 1, 3])
        offsets = torch.tensor([0, 2, 3, 5])
        scale_grad_by_freq = False
        sparse = False
        per_sample_weights = None
        include_last_offset = False
        padding_idx = -1

        (
            output,
            offset2bag,
            bag_size,
            maximum_indices,
        ) = torch.ops.aten._embedding_bag.default(
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        )
        grad = torch.randn_like(output)

        # Call the function with example inputs
        grad_weight = torch.ops.aten._embedding_bag_dense_backward.default(
            grad,
            indices,
            offset2bag,
            bag_size,
            maximum_indices,
            weight.size(0),
            scale_grad_by_freq,
            mode,
            per_sample_weights,
            padding_idx,
        )
        meta_grad_weight = torch.ops.aten._embedding_bag_dense_backward.default(
            grad.to("meta"),
            indices.to("meta"),
            offset2bag.to("meta"),
            bag_size.to("meta"),
            maximum_indices.to("meta"),
            weight.size(0),
            scale_grad_by_freq,
            mode,
            per_sample_weights,
            padding_idx,
        )
        self.assertEqual(grad_weight.to("meta"), meta_grad_weight)

    def test_segment_reduce_backward(self):
        grad = torch.ones(16, dtype=torch.float)
        output = torch.ones(16, dtype=torch.float)
        data = torch.ones(16, dtype=torch.float)
        reduce_str = "max"
        lengths = torch.ones(16, dtype=torch.long)

        out = torch.ops.aten._segment_reduce_backward(
            grad, output, data, reduce_str, lengths=lengths
        )
        out_meta = torch.ops.aten._segment_reduce_backward(
            grad.to(device="meta"),
            output.to(device="meta"),
            data.to(device="meta"),
            reduce_str,
            lengths=lengths.to(device="meta"),
        )
        self.assertEqual(out.shape, out_meta.shape)
        self.assertEqual(out.stride(), out_meta.stride())
        self.assertEqual(out.dtype, out_meta.dtype)
        self.assertEqual(out.layout, out_meta.layout)

        # noncontiguous
        grad = torch.ones(16, 2, dtype=torch.float)[:, 1]
        data = torch.ones(16, 2, dtype=torch.float)[:, 1]
        out = torch.ops.aten._segment_reduce_backward(
            grad, output, data, reduce_str, lengths=lengths
        )
        out_meta = torch.ops.aten._segment_reduce_backward(
            grad.to(device="meta"),
            output.to(device="meta"),
            data.to(device="meta"),
            reduce_str,
            lengths=lengths.to(device="meta"),
        )
        self.assertEqual(out.shape, out_meta.shape)
        self.assertEqual(out.stride(), out_meta.stride())
        self.assertEqual(out.dtype, out_meta.dtype)
        self.assertEqual(out.layout, out_meta.layout)

    def test_embedding_bag_dense_backward_per_sample_weights(self):
        weight = torch.randn(4, 3, requires_grad=True)
        indices = torch.tensor([1, 0, 2, 1, 3])
        offsets = torch.tensor([0, 2, 3, 5])
        scale_grad_by_freq = False
        sparse = False
        mode = 0
        per_sample_weights = torch.randn(5, requires_grad=True)
        include_last_offset = False
        padding_idx = -1

        (
            output,
            offset2bag,
            bag_size,
            maximum_indices,
        ) = torch.ops.aten._embedding_bag.default(
            weight,
            indices,
            offsets,
            scale_grad_by_freq,
            mode,
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx,
        )
        grad = torch.randn_like(output)

        # Call the function with example inputs
        grad_weight = torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
            grad, weight, indices, offsets, offset2bag, mode, padding_idx
        )
        meta_grad_weight = (
            torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
                grad.to("meta"),
                weight.to("meta"),
                indices.to("meta"),
                offsets.to("meta"),
                offset2bag.to("meta"),
                mode,
                padding_idx,
            )
        )
        self.assertEqual(grad_weight.to("meta"), meta_grad_weight)

    # opinfo test is using aten.fill_, it's not testing aten.fill
    @onlyOn(["cuda", "xpu"])
    def test_fill_stride(self):
        to_meta = MetaConverter()
        sample_args = [torch.rand(2, 2, 2, 2), 1.0]

        for args in get_strided_args(sample_args):
            meta_args = to_meta(args)
            ref_out = torch.ops.aten.fill(*args)
            meta_out = torch.ops.aten.fill(*meta_args)
            self.assertEqual(ref_out.size(), meta_out.size())
            self.assertEqual(ref_out.stride(), meta_out.stride())

    def test_map_location_deserialize(self):
        import io

        t = torch.rand(10)
        b = io.BytesIO()

        torch.save(t, b)
        b.seek(0)
        r = torch.load(b, map_location=torch.device("meta"))
        self.assertEqual(r.device.type, "meta")
        self.assertEqual(r.shape, t.shape)
        self.assertEqual(r.dtype, t.dtype)
        self.assertEqual(r.storage().data_ptr(), 0)

    def test_embedding_bag_byte_prepack(self):
        batch_size = 10
        num_embeddings = 80
        embedding_dim = [128, 256, 512]
        res_shape = [[batch_size, num_embeddings, ed + 8] for ed in embedding_dim]
        for ed, rs in zip(embedding_dim, res_shape):
            weight = torch.randn(batch_size, num_embeddings, ed, dtype=torch.float32)
            res = torch.ops.quantized.embedding_bag_byte_prepack(
                weight.to(device="meta")
            )
            self.assertEqual(res.shape, rs)
            self.assertEqual(res.dtype, torch.float32)
            self.assertEqual(res.untyped_storage().data_ptr(), 0)

    def test_embedding_bag_byte_unpack(self):
        batch_size = 10
        num_embeddings = 80
        embedding_dim = [128, 256, 512]
        res_shape = [[batch_size, num_embeddings, ed] for ed in embedding_dim]
        for ed, rs in zip(embedding_dim, res_shape):
            packed_weight = torch.randn(
                batch_size, num_embeddings, ed + 8, dtype=torch.float32
            )
            res = torch.ops.quantized.embedding_bag_byte_unpack(
                packed_weight.to(device="meta")
            )
            self.assertEqual(res.shape, rs)
            self.assertEqual(res.dtype, torch.float32)
            self.assertEqual(res.untyped_storage().data_ptr(), 0)

    def test_index_select_out(self):
        def f():
            input = torch.randn([8, 16], device="meta")
            index = torch.tensor([2, 1, 6, 7, 3, 1, 7, 5, 6, 7], device="meta")
            out = torch.empty([10, 16], device="meta")
            return torch.index_select(input=input, dim=0, index=index, out=out)

        with enable_python_dispatcher():
            out = f()
            self.assertEqual(out.shape, [10, 16])

    def test_local_scalar_dense_call(self):
        with self.assertRaisesRegex(RuntimeError, "cannot be called on meta tensors"):
            meta_tensor = torch.randn(1, device="meta")
            meta_tensor.item()

    def test_triangular_solve_out(self):
        # Get what's the expected output for the given example.
        A = torch.randn(2, 2).triu()
        b = torch.randn(2, 3)
        out = torch.triangular_solve(b, A)

        # Call the function again, transforming every tensor input (including the out tensor)
        # into a meta tensor.
        meta_out = tree_map_only(torch.Tensor, lambda t: t.to("meta"), out)
        torch.triangular_solve(b.to("meta"), A.to("meta"), out=meta_out)

        self.assertEqual(out[0].shape, meta_out[0].shape)
        self.assertEqual(out[0].dtype, meta_out[0].dtype)

        self.assertEqual(out[1].shape, meta_out[1].shape)
        self.assertEqual(out[1].dtype, meta_out[1].dtype)

    def test_meta_consistency_out_dtype_mismatch_pow_Tensor_Scalar(self):
        S = (5,)

        def run(device):
            a = torch.rand(S, device=device, dtype=torch.float32)
            b = 2
            out = torch.empty(S, device=device, dtype=torch.float64)

            try:
                torch.pow(a, b, out=out)
            except Exception as e:
                return e

        cpu_err = run("cpu")
        meta_err = run("meta")

        if cpu_err is None and meta_err is not None:
            raise RuntimeError("cpu didn't fail, but meta did.") from meta_err
        elif cpu_err is not None and meta_err is None:
            raise RuntimeError("cpu failed, but meta didn't.") from cpu_err

    def test_nonzero(self):
        t = torch.randn(2, 3, 4, device="meta")
        with exp_config.patch(meta_nonzero_assume_all_nonzero=True):
            nz = t.nonzero()
        self.assertEqual(nz.dtype, torch.int64)
        self.assertEqual(nz.device.type, "meta")
        self.assertEqual(nz.shape, torch.Size([24, 3]))
        self.assertEqual(nz.stride(), torch.Size([1, 24]))

    def test_stride_for_index_Tensor(self):
        from torch._subclasses import FakeTensorMode

        x = torch.randn((24, 16, 32, 32)).to(memory_format=torch.channels_last)
        x = x.view(2, 12, 16, 32, 32)

        i1 = torch.arange(2).unsqueeze(-1)
        i2 = torch.argsort(torch.rand(2, 12), dim=-1)[:, :3]

        out = x[i1, i2]

        mode = FakeTensorMode()
        with mode:
            f_x = mode.from_tensor(x)
            f_i1 = mode.from_tensor(i1)
            f_i2 = mode.from_tensor(i2)
            f_out = f_x[f_i1, f_i2]

        self.assertEqual(out.stride(), f_out.stride())

    @parametrize("in_dtype", [torch.float32, torch.float16])
    @parametrize("bias_dtype", [torch.float32, torch.float16, None])
    def test_mixed_dtype_for_native_layer_norm_backward(self, in_dtype, bias_dtype):
        if in_dtype == torch.float16 and bias_dtype == torch.float32:
            self.skipTest(
                f"not supported input dtype is {in_dtype} and bias dtype is {bias_dtype}"
            )
        device = "meta"

        def fn(input, weight, bias, need_grad_input):
            outputs = torch.nn.functional.layer_norm(
                input, input.shape[-1:], weight, bias
            )
            grad_outs = torch.ones_like(outputs)
            grad_ins = torch.autograd.grad(outputs, need_grad_input, grad_outs)
            return grad_ins

        input = torch.randn(
            [4, 8, 5], dtype=in_dtype, device=device, requires_grad=True
        )
        need_grad_input = [input]

        if bias_dtype:
            weight = torch.randn(
                [5], dtype=bias_dtype, device=device, requires_grad=True
            )
            bias = torch.randn([5], dtype=bias_dtype, device=device, requires_grad=True)
            need_grad_input.append(weight)
            need_grad_input.append(bias)
        else:
            weight = None
            bias = None

        outs = fn(input, weight, bias, need_grad_input)
        out_dtype = [t.dtype for t in outs]
        if bias_dtype:
            self.assertEqual(out_dtype, [in_dtype, bias_dtype, bias_dtype])
        else:
            self.assertEqual(
                out_dtype,
                [
                    in_dtype,
                ],
            )


instantiate_device_type_tests(TestMeta, globals(), only_for="xpu", allow_xpu=True)


def print_op_str_if_not_supported(op_str):
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


if __name__ == "__main__":
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
