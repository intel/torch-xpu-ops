# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

_dtype_pairs = [
    (torch.float16, torch.float16),
    (torch.float16, torch.float32),
    (torch.float16, torch.float64),
    (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float32),
    (torch.bfloat16, torch.float64),
    (torch.bfloat16, torch.float16),
    (torch.float32, torch.float32),
    (torch.float32, torch.float64),
    (torch.float32, torch.float16),
    (torch.float32, torch.bfloat16),
]


class TestAddcOutDtype(TestCase):
    def _check_device_dtypes(self, device, *dtypes):
        if any(dtype in (torch.float64, torch.complex128) for dtype in dtypes):
            if not torch.xpu.get_device_properties(device).has_fp64:
                self.skipTest("FP64 is not supported on this XPU")

    @parametrize("op_name", ["addcmul", "addcdiv"])
    @parametrize("dtype_pair", _dtype_pairs)
    @parametrize("layout", ["contiguous", "strided", "broadcast", "out_strided"])
    def test_out_preserves_common_dtype(self, device, op_name, dtype_pair, layout):
        dtype, out_dtype = dtype_pair
        self._check_device_dtypes(device, dtype, out_dtype)
        op = getattr(torch, op_name)
        tiny = {torch.float16: 2**-6, torch.bfloat16: 2**-4, torch.float32: 2**-12}[
            dtype
        ]
        large = 60000.0 if dtype == torch.float16 else 3e38
        self_t = torch.tensor([[1.0, 2.0, -1.0, 1.0]] * 2, dtype=dtype)
        t1 = torch.tensor([[tiny, 3.0, -tiny, large]] * 2, dtype=dtype)
        t2 = torch.tensor(
            [
                [tiny, 5.0, tiny, 4.0]
                if op_name == "addcmul"
                else [1 / tiny, 4.0, 1 / tiny, 0.5]
            ]
            * 2,
            dtype=dtype,
        )
        if layout == "broadcast":
            self_t, t1 = self_t[:1], t1[:1]
            t2 = t2[:, :1]
        expected = op(self_t, t1, t2).to(out_dtype)
        args = [t.to(device) for t in (self_t, t1, t2)]
        if layout == "strided":
            args = [torch.stack((t, t), dim=-1)[..., 0] for t in args]
            self.assertTrue(all(not t.is_contiguous() for t in args))
        out = torch.empty(
            (4, 2) if layout == "out_strided" else (2, 4),
            dtype=out_dtype,
            device=device,
        )
        if layout == "out_strided":
            out = out.t()
        pointer, strides = out.data_ptr(), out.stride()
        actual = op(*args, out=out)
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)
        self.assertEqual(actual.dtype, out_dtype)
        self.assertEqual(actual.data_ptr(), pointer)
        self.assertEqual(actual.stride(), strides)

    @parametrize("dtype_pair", _dtype_pairs)
    @parametrize("scalar_dtype", ["same", "double"])
    @parametrize("layout", ["contiguous", "strided", "broadcast", "out_strided"])
    def test_cpu_scalar_out_preserves_common_dtype(
        self, device, dtype_pair, scalar_dtype, layout
    ):
        dtype, out_dtype = dtype_pair
        self._check_device_dtypes(device, dtype, out_dtype)
        tiny = {torch.float16: 2**-6, torch.bfloat16: 2**-4, torch.float32: 2**-12}[
            dtype
        ]
        self_t = torch.tensor([1.0, 2.0, -1.0], dtype=dtype)
        t1 = torch.tensor([tiny, 3.0, -tiny], dtype=dtype)
        t2 = torch.tensor(
            tiny, dtype=dtype if scalar_dtype == "same" else torch.float64
        )
        self_t, t1 = self_t.repeat(2, 1), t1.repeat(2, 1)
        if layout == "broadcast":
            t1 = t1[:1]
        expected = torch.addcmul(self_t, t1, t2).to(out_dtype)
        args = [t.to(device) for t in (self_t, t1)]
        if layout == "strided":
            args = [torch.stack((t, t), dim=-1)[..., 0] for t in args]
        out = torch.empty(
            (3, 2) if layout == "out_strided" else (2, 3),
            dtype=out_dtype,
            device=device,
        )
        if layout == "out_strided":
            out = out.t()
        actual = torch.addcmul(*args, t2, out=out)
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize(
        "op_scalar", [("addcmul", False), ("addcmul", True), ("addcdiv", False)]
    )
    @parametrize("out_dtype", [torch.float32, torch.float64])
    @parametrize("strided", [False, True])
    def test_half_rounding_boundaries(self, device, op_scalar, out_dtype, strided):
        op_name, cpu_scalar = op_scalar
        self._check_device_dtypes(device, out_dtype)
        op = getattr(torch, op_name)
        self_t = torch.tensor(
            [1, 1, -1, 0, 0, 65504, -65504, 2**-14], dtype=torch.float16
        )
        t1 = torch.tensor(
            [2**-11, 3 * 2**-11, -(2**-11), 2**-24, -(2**-24), 16, -16, 2**-24],
            dtype=torch.float16,
        )
        t2 = (
            torch.tensor(1.0, dtype=torch.float16)
            if cpu_scalar
            else torch.ones_like(t1)
        )
        # Halfway cases exercise ties-to-even, plus subnormals and overflow.
        expected = op(self_t, t1, t2).to(out_dtype)
        args = [t.to(device) for t in (self_t, t1)]
        if strided:
            args = [torch.stack((t, t), dim=-1)[..., 0] for t in args]
        actual = op(
            *args,
            t2 if cpu_scalar else t2.to(device),
            out=torch.empty(8, dtype=out_dtype, device=device),
        )
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("out_kind", ["same", "float", "double"])
    @parametrize("strided", [False, True])
    def test_cpu_scalar_casts_to_common_dtype(self, device, dtype, out_kind, strided):
        scalar = {
            torch.float16: 1.0006,
            torch.bfloat16: 1.005,
            torch.float32: 1 + 2**-24 + 2**-26,
        }[dtype]
        a = torch.zeros(4, dtype=dtype)
        b = torch.full_like(a, 3)
        c = torch.tensor(scalar, dtype=torch.float64)
        out_dtype = {"same": dtype, "float": torch.float32, "double": torch.float64}[
            out_kind
        ]
        self._check_device_dtypes(device, out_dtype)
        expected = torch.addcmul(a, b, c).to(out_dtype)
        args = [t.to(device) for t in (a, b)]
        if strided:
            args = [torch.stack((t, t), dim=-1)[..., 0] for t in args]
        actual = torch.addcmul(
            *args, c, out=torch.empty(4, dtype=out_dtype, device=device)
        )
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize(
        "op_position",
        [(op, False, pos) for op in ("addcmul", "addcdiv") for pos in range(3)]
        + [("addcmul", True, pos) for pos in range(2)],
    )
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("scalar_dtype", [torch.float32, torch.float64])
    @parametrize("out_kind", ["same", "other", "float", "double"])
    @parametrize("numel", [1, 8])
    @parametrize("strided", [False, True])
    def test_gpu_scalar_casts_to_common_dtype(
        self, device, op_position, dtype, scalar_dtype, out_kind, numel, strided
    ):
        op_name, cpu_tensor2, scalar_position = op_position
        op = getattr(torch, op_name)
        args = [
            torch.full((numel,), -1.0, dtype=dtype),
            torch.ones(numel, dtype=dtype),
            torch.tensor(1.0, dtype=dtype)
            if cpu_tensor2
            else torch.ones(numel, dtype=dtype),
        ]
        args[scalar_position] = torch.tensor(
            -1 + 2**-12 if scalar_position == 0 else 1 + 2**-12,
            dtype=scalar_dtype,
        )
        out_dtype = {
            "same": dtype,
            "other": torch.bfloat16 if dtype == torch.float16 else torch.float16,
            "float": torch.float32,
            "double": torch.float64,
        }[out_kind]
        self._check_device_dtypes(device, scalar_dtype, out_dtype)
        reference = op(*args)
        self.assertEqual(reference.dtype, dtype)
        expected = reference.to(out_dtype)
        args = [
            t if cpu_tensor2 and i == 2 else t.to(device) for i, t in enumerate(args)
        ]
        if strided:
            args = [torch.stack((t, t), dim=-1)[..., 0] if t.dim() else t for t in args]
        out = torch.empty(numel * (2 if strided else 1), dtype=out_dtype, device=device)
        if strided:
            out = out[::2]
        actual = op(*args, out=out)
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize(
        "op_position",
        [(op, False, pos) for op in ("addcmul", "addcdiv") for pos in range(3)]
        + [("addcmul", True, pos) for pos in range(2)],
    )
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("integer_dtype", [torch.int32, torch.int64])
    @parametrize("out_kind", ["same", "other", "float", "double"])
    @parametrize("strided", [False, True])
    def test_integer_input_casts_to_common_dtype(
        self, device, op_position, dtype, integer_dtype, out_kind, strided
    ):
        op_name, cpu_tensor2, integer_position = op_position
        op = getattr(torch, op_name)
        args = [
            torch.full((8,), -2048.0, dtype=dtype),
            torch.full((8,), 2048.0 if integer_position == 0 else 1.0, dtype=dtype),
            torch.tensor(1.0, dtype=dtype)
            if cpu_tensor2
            else torch.ones(8, dtype=dtype),
        ]
        args[integer_position] = torch.full(
            (8,), -2049 if integer_position == 0 else 2049, dtype=integer_dtype
        )
        if op_name == "addcdiv" and integer_position == 2:
            args[0].fill_(-1)
            args[1].fill_(2048)
        out_dtype = {
            "same": dtype,
            "other": torch.bfloat16 if dtype == torch.float16 else torch.float16,
            "float": torch.float32,
            "double": torch.float64,
        }[out_kind]
        self._check_device_dtypes(device, out_dtype)
        reference = op(*args)
        self.assertEqual(reference.dtype, dtype)
        args = [
            t if cpu_tensor2 and i == 2 else t.to(device) for i, t in enumerate(args)
        ]
        if strided:
            args = [torch.stack((t, t), dim=-1)[..., 0] if t.dim() else t for t in args]
        actual = op(*args, out=torch.empty(8, dtype=out_dtype, device=device))
        self.assertEqual(actual.cpu(), reference.to(out_dtype), atol=0, rtol=0)

    @parametrize("case", ["overflow", "zero_times_overflow", "underflow_divisor"])
    @parametrize("scalar_dtype", [torch.float32, torch.float64, torch.bfloat16])
    @parametrize(
        "out_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    )
    @parametrize("strided", [False, True])
    def test_gpu_scalar_cast_before_arithmetic(
        self, device, case, scalar_dtype, out_dtype, strided
    ):
        self._check_device_dtypes(device, scalar_dtype, out_dtype)
        if case == "underflow_divisor":
            op, a, b, c, value = torch.addcdiv, 0.0, 2**-24, 2**-30, 1
        elif case == "zero_times_overflow":
            op, a, b, c, value = torch.addcmul, 1.0, 1.0, 65536.0, 0
        else:
            op, a, b, c, value = torch.addcmul, -65504.0, 1.0, 65536.0, 1
        args = [torch.full((8,), x, dtype=torch.float16) for x in (a, b)]
        args.append(torch.tensor(c, dtype=scalar_dtype))
        expected = op(*args, value=value).to(out_dtype)
        args = [t.to(device) for t in args]
        if strided:
            args = [torch.stack((t, t), dim=-1)[..., 0] if t.dim() else t for t in args]
        actual = op(
            *args, value=value, out=torch.empty(8, dtype=out_dtype, device=device)
        )
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize("cpu_scalar", [False, True])
    def test_half_overflow_cast_before_wide_store(self, device, cpu_scalar):
        self_t = torch.tensor([1.0, 2.0], dtype=torch.float16)
        t1 = torch.tensor([60000.0, 3.0], dtype=torch.float16)
        t2 = torch.tensor(4.0 if cpu_scalar else [4.0, 5.0], dtype=torch.float16)
        expected = torch.addcmul(self_t, t1, t2).float()
        actual = torch.addcmul(
            self_t.to(device),
            t1.to(device),
            t2 if cpu_scalar else t2.to(device),
            out=torch.empty(2, dtype=torch.float32, device=device),
        )
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize("op_name", ["addcmul", "addcdiv"])
    @parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("inplace", [False, True])
    def test_promote_inputs_before_narrow_output(
        self, device, op_name, out_dtype, inplace
    ):
        op = getattr(torch, op_name)
        self_t = torch.tensor([1.0, 2.0], dtype=out_dtype)
        t1 = torch.tensor([131072.0, 3.0], dtype=torch.float32)
        t2 = torch.tensor(
            [0.5, 4.0] if op_name == "addcmul" else [2.0, 4.0], dtype=torch.float32
        )
        expected = op(self_t, t1, t2, value=0.125).to(out_dtype)
        target = self_t.to(device)
        if inplace:
            actual = getattr(target, op_name + "_")(
                t1.to(device), t2.to(device), value=0.125
            )
        else:
            actual = op(
                target,
                t1.to(device),
                t2.to(device),
                value=0.125,
                out=torch.empty_like(target),
            )
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize("op_name", ["addcmul", "addcdiv"])
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @parametrize("wide_out", [False, True])
    def test_value_zero_preserves_non_finite_inputs(
        self, device, op_name, dtype, wide_out
    ):
        op = getattr(torch, op_name)
        self_t = torch.ones(3, dtype=dtype)
        t1 = torch.tensor([float("nan"), float("inf"), float("-inf")], dtype=dtype)
        t2 = torch.zeros(3, dtype=dtype)
        out_dtype = torch.float64 if wide_out else dtype
        self._check_device_dtypes(device, out_dtype)
        expected = op(self_t, t1, t2, value=0).to(out_dtype)
        actual = op(
            self_t.to(device),
            t1.to(device),
            t2.to(device),
            value=0,
            out=torch.empty(3, dtype=out_dtype, device=device),
        )
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize(
        "op_scalar", [("addcmul", False), ("addcmul", True), ("addcdiv", False)]
    )
    @parametrize("dtype", [torch.float32, torch.complex64])
    def test_complex_output(self, device, op_scalar, dtype):
        self._check_device_dtypes(device, torch.complex128)
        op_name, cpu_scalar = op_scalar
        op = getattr(torch, op_name)
        self_t = torch.tensor([1.0, 2.0], dtype=dtype)
        if dtype.is_complex:
            self_t += 1j
        t1 = torch.tensor([2.0, 3.0], dtype=dtype)
        t2 = torch.tensor(4.0 if cpu_scalar else [4.0, 4.0], dtype=dtype)
        expected = op(self_t, t1, t2).to(torch.complex128)
        actual = op(
            self_t.to(device),
            t1.to(device),
            t2 if cpu_scalar else t2.to(device),
            out=torch.empty(2, dtype=torch.complex128, device=device),
        )
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize("out_dtype", [torch.int64, torch.float32])
    @parametrize("cpu_scalar", [False, True])
    def test_integer_common_dtype(self, device, out_dtype, cpu_scalar):
        self_t = torch.tensor([1, 2], dtype=torch.int8)
        t1 = torch.tensor([120, 3], dtype=torch.int8)
        t2 = torch.tensor(2 if cpu_scalar else [2, 2], dtype=torch.int8)
        expected = torch.addcmul(self_t, t1, t2).to(out_dtype)
        actual = torch.addcmul(
            self_t.to(device),
            t1.to(device),
            t2 if cpu_scalar else t2.to(device),
            out=torch.empty(2, dtype=out_dtype, device=device),
        )
        self.assertEqual(actual.cpu(), expected, atol=0, rtol=0)

    @parametrize("op_name", ["addcmul", "addcdiv"])
    def test_empty_wide_output(self, device, op_name):
        args = [
            torch.empty((0, 3), dtype=torch.float16, device=device) for _ in range(3)
        ]
        out = torch.empty((0, 3), dtype=torch.float32, device=device)
        actual = getattr(torch, op_name)(*args, out=out)
        self.assertEqual(actual.shape, (0, 3))
        self.assertEqual(actual.dtype, torch.float32)


instantiate_device_type_tests(
    TestAddcOutDtype, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
