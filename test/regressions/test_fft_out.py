# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
#
# The XPU MKL FFT out variants write into the caller-provided output whenever
# that is safe. _exec_fft rewrites its destination metadata with
# resize_/as_strided_, so the direct-write path must be restricted to the cases
# where the resulting layout still matches a contiguous output. These tests pin
# both the returned values and the output tensor identity/layout.

import itertools

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase

_layout_cases = [
    ((2, 3, 4, 5)[-ndim:], dims)
    for ndim in range(1, 5)
    for count in range(1, ndim + 1)
    for dims in itertools.combinations(range(ndim), count)
]


class TestFftOut(TestCase):
    @parametrize(
        "shape,dims",
        _layout_cases,
        name_fn=lambda shape, dims: (
            f"shape_{'x'.join(map(str, shape))}_dims_{'_'.join(map(str, dims))}"
        ),
    )
    @parametrize("op", ("c2c", "r2c", "c2r"))
    def test_fft_out_preserves_contiguous_layout(self, device, shape, dims, op):
        # _exec_fft may permute the functional result's strides. A contiguous
        # caller-provided output must retain its own layout for every dim set.
        if op == "c2c":
            x = torch.randn(shape, dtype=torch.complex128, device=device)
            cpu = x.cpu()
            args = (dims, 0, True)
        else:
            x = torch.randn(shape, dtype=torch.float64, device=device)
            cpu = x.cpu()
            if op == "r2c":
                args = (dims, 0, True)
            else:
                last_size = shape[dims[-1]]
                x = torch.ops.aten._fft_r2c.default(x, dims, 0, True)
                cpu = torch.ops.aten._fft_r2c.default(cpu, dims, 0, True)
                args = (dims, 0, last_size)

        fft = getattr(torch.ops.aten, f"_fft_{op}")
        expected = fft.default(x, *args)
        # Anchor the shared functional/out implementation to an independent CPU
        # result so an error common to both paths cannot hide behind equality.
        self.assertEqual(expected.cpu(), fft.default(cpu, *args))
        # empty_like would inherit the functional result's permuted strides.
        out = torch.empty(expected.shape, dtype=expected.dtype, device=device)
        stride = out.stride()
        result = fft.out(x, *args, out=out)
        self.assertIs(result, out)
        self.assertEqual(out.stride(), stride)
        self.assertEqual(out, expected)

    @parametrize(
        "dims",
        ((2,), (1, 2), (0, 1, 2)),
        name_fn=lambda dims: f"dims_{'_'.join(map(str, dims))}",
    )
    def test_fft_out_noncontiguous_input(self, device, dims):
        # The direct-write predicate inspects the batch strides of the input, so
        # a transposed input has to keep both its values and the out layout.
        x = torch.randn(4, 3, 8, dtype=torch.complex128, device=device).transpose(0, 1)
        expected = torch.ops.aten._fft_c2c.default(x, dims, 0, True)
        out = torch.empty(expected.shape, dtype=expected.dtype, device=device)
        stride = out.stride()
        result = torch.ops.aten._fft_c2c.out(x, dims, 0, True, out=out)
        self.assertIs(result, out)
        self.assertEqual(out.stride(), stride, f"dims={dims}")
        self.assertEqual(out, expected, msg=f"dims={dims}")

    @parametrize("onesided", (True, False))
    def test_fft_r2c_out_preserves_layout_for_size_one_input_strides(
        self, device, onesided
    ):
        # A size-one dimension may have an arbitrary stride while the tensor is
        # still contiguous. The out path must not propagate that input layout to
        # the caller-provided output.
        source = torch.randn(2, 1, 4, device=device)
        x = torch.empty_strided((2, 1, 4), (4, 999, 1), device=device)
        x.copy_(source)
        self.assertTrue(x.is_contiguous())

        expected = torch.ops.aten._fft_r2c.default(x, [2], 0, onesided)
        out = torch.empty(expected.shape, dtype=expected.dtype, device=device)
        stride = out.stride()
        result = torch.ops.aten._fft_r2c.out(x, [2], 0, onesided, out=out)
        self.assertIs(result, out)
        self.assertEqual(out.stride(), stride, f"onesided={onesided}")
        self.assertEqual(out, expected, msg=f"onesided={onesided}")

    def test_fft_out_noncontiguous_output(self, device):
        complex_xpu = torch.randn(2, 3, 4, 5, dtype=torch.complex64, device=device)
        expected = torch.ops.aten._fft_c2c.default(complex_xpu, [2, 3], 0, True)

        out = torch.empty(5, 4, 3, 2, dtype=torch.complex64, device=device).permute(
            3, 2, 1, 0
        )
        stride = out.stride()
        result = torch.ops.aten._fft_c2c.out(complex_xpu, [2, 3], 0, True, out=out)
        self.assertIs(result, out)
        self.assertEqual(out.stride(), stride)
        self.assertEqual(out, expected)

    def test_fft_out_preserves_size_one_strides(self, device):
        xc = torch.randn(2, 1, 4, dtype=torch.complex64, device=device)
        xr = torch.randn(2, 1, 4, device=device)

        c2c_expected = torch.ops.aten._fft_c2c.default(xc, [2], 0, True)
        c2c_out = torch.empty_strided(
            c2c_expected.shape, (4, 99, 1), dtype=torch.complex64, device=device
        )
        torch.ops.aten._fft_c2c.out(xc, [2], 0, True, out=c2c_out)
        self.assertEqual(c2c_out.stride(), (4, 99, 1))
        self.assertEqual(c2c_out, c2c_expected)

        r2c_expected = torch.ops.aten._fft_r2c.default(xr, [2], 0, True)
        r2c_out = torch.empty_strided(
            r2c_expected.shape, (3, 99, 1), dtype=torch.complex64, device=device
        )
        torch.ops.aten._fft_r2c.out(xr, [2], 0, True, out=r2c_out)
        self.assertEqual(r2c_out.stride(), (3, 99, 1))
        self.assertEqual(r2c_out, r2c_expected)

        c2r_expected = torch.ops.aten._fft_c2r.default(r2c_expected, [2], 0, 4)
        c2r_out = torch.empty_strided(
            c2r_expected.shape, (4, 99, 1), dtype=torch.float32, device=device
        )
        torch.ops.aten._fft_c2r.out(r2c_expected, [2], 0, 4, out=c2r_out)
        self.assertEqual(c2r_out.stride(), (4, 99, 1))
        self.assertEqual(c2r_out, c2r_expected)

    def test_fft_out_aliased_input(self, device):
        x = torch.randn(2, 3, 4, dtype=torch.complex64, device=device)
        expected = torch.ops.aten._fft_c2c.default(x, [2], 0, True)
        result = torch.ops.aten._fft_c2c.out(x, [2], 0, True, out=x)
        self.assertIs(result, x)
        self.assertEqual(x, expected)

    def test_fft_out_empty_dim(self, device):
        # An empty dim list short-circuits to a clone in the functional variant,
        # so the out variants must reach it too rather than the impl, whose
        # r2c form asserts the dim list is non-empty.
        x = torch.randn(2, 3, dtype=torch.complex64, device=device)
        out = torch.empty_like(x)
        result = torch.ops.aten._fft_c2c.out(x, [], 0, True, out=out)
        self.assertIs(result, out)
        self.assertEqual(out, x)

        # The clone keeps the input dtype, so r2c reports a real result here.
        real_x = torch.randn(2, 3, device=device)
        functional = torch.ops.aten._fft_r2c.default(real_x, [], 0, True)
        out = torch.empty(0, dtype=functional.dtype, device=device)
        result = torch.ops.aten._fft_r2c.out(real_x, [], 0, True, out=out)
        self.assertIs(result, out)
        self.assertEqual(out, functional)

        functional = torch.ops.aten._fft_c2r.default(x, [], 0, 4)
        out = torch.empty(0, dtype=functional.dtype, device=device)
        result = torch.ops.aten._fft_c2r.out(x, [], 0, 4, out=out)
        self.assertIs(result, out)
        self.assertEqual(out, functional)

    @parametrize(
        "dtype,expected_dtype",
        (
            (torch.float32, torch.complex64),
            (torch.float64, torch.complex128),
            (torch.half, torch.complex32),
            (torch.bfloat16, torch.complex64),
        ),
    )
    def test_fft_r2c_out_dtype_promotion(self, device, dtype, expected_dtype):
        # promote_fft_input promotes half/bfloat16 to float. The functional
        # variant must keep reporting the same output dtype as before the out
        # variants started sharing its implementation.
        x = torch.randn(8, 16, device=device, dtype=dtype)
        functional = torch.ops.aten._fft_r2c.default(x, [1], 0, True)
        self.assertEqual(functional.dtype, expected_dtype)

        out = torch.empty(functional.shape, dtype=expected_dtype, device=device)
        result = torch.ops.aten._fft_r2c.out(x, [1], 0, True, out=out)
        self.assertIs(result, out)
        self.assertEqual(out.dtype, expected_dtype)
        self.assertEqual(out, functional)

    @parametrize("normalization", (0, 1, 2))
    def test_fft_out_casts_from_the_functional_result_dtype(
        self, device, normalization
    ):
        # The out= contract is functional-result-then-copy: an out tensor with
        # a wider dtype must contain the low-precision functional result cast
        # to that dtype, rather than a result computed directly at the wider
        # transform precision.
        xc = torch.randn(4, 16, dtype=torch.complex64, device=device).to(
            torch.complex32
        )
        xr = torch.randn(4, 16, dtype=torch.float16, device=device)

        c2c_expected = torch.ops.aten._fft_c2c.default(xc, [1], normalization, True)
        c2c_out = torch.empty_like(c2c_expected, dtype=torch.complex64)
        torch.ops.aten._fft_c2c.out(xc, [1], normalization, True, out=c2c_out)
        self.assertEqual(
            c2c_out,
            c2c_expected.to(c2c_out.dtype),
            msg=f"c2c normalization={normalization}",
        )

        r2c_expected = torch.ops.aten._fft_r2c.default(xr, [1], normalization, True)
        r2c_out = torch.empty_like(r2c_expected, dtype=torch.complex64)
        torch.ops.aten._fft_r2c.out(xr, [1], normalization, True, out=r2c_out)
        self.assertEqual(
            r2c_out,
            r2c_expected.to(r2c_out.dtype),
            msg=f"r2c normalization={normalization}",
        )

        c2r_expected = torch.ops.aten._fft_c2r.default(xc, [1], normalization, 30)
        c2r_out = torch.empty_like(c2r_expected, dtype=torch.float32)
        torch.ops.aten._fft_c2r.out(xc, [1], normalization, 30, out=c2r_out)
        self.assertEqual(
            c2r_out,
            c2r_expected.to(c2r_out.dtype),
            msg=f"c2r normalization={normalization}",
        )

    @parametrize("op", ("c2c", "c2r"))
    @parametrize("normalization", (1, 2))
    def test_fft_half_normalization(self, device, op, normalization):
        # Complex-half c2c/c2r must cast the promoted FFT result before applying
        # normalization, including when the functional wrapper owns that step.
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(2, 3, dtype=torch.complex64, generator=generator).to(
            device=device, dtype=torch.complex32
        )
        if op == "c2c":
            dims, extra, size, dtype = [1], True, 3, torch.complex32
        else:
            dims, extra, size, dtype = [0, 1], 4, 8, torch.float16
        fft = getattr(torch.ops.aten, f"_fft_{op}")
        unnormalized = fft.default(x, dims, 0, extra)
        scale = 1 / (size**0.5 if normalization == 1 else size)
        expected = unnormalized.clone().mul_(scale)
        got = fft.default(x, dims, normalization, extra)
        self.assertEqual(got.dtype, dtype)
        self.assertEqual(got, expected, rtol=0, atol=0)

        out = torch.empty(got.shape, dtype=dtype, device=device)
        fft.out(x, dims, normalization, extra, out=out)
        self.assertEqual(out, got)

    @parametrize(
        "ndim,dims", ((1, (0,)), (2, (1,)), (2, (0, 1)), (3, (2,)), (3, (0, 1, 2)))
    )
    def test_fft_r2c_out_twosided(self, device, ndim, dims):
        # onesided=False fills the second half using conjugate symmetry.
        shape = (2, 3, 4)[-ndim:]
        x = torch.randn(*shape, dtype=torch.float64, device=device)
        expected = torch.fft.fftn(x.to(torch.complex128), dim=dims)
        msg = f"ndim={ndim} dims={dims}"

        functional = torch.ops.aten._fft_r2c.default(x, dims, 0, False)
        self.assertEqual(functional, expected, msg=msg)

        out = torch.empty(functional.shape, dtype=functional.dtype, device=device)
        stride = out.stride()
        result = torch.ops.aten._fft_r2c.out(x, dims, 0, False, out=out)
        self.assertIs(result, out)
        self.assertEqual(out.stride(), stride, f"stride {msg}")
        self.assertEqual(out, expected, msg=msg)

    @parametrize("dtype", (torch.complex64, torch.complex32))
    def test_fft_c2r_does_not_mutate_input(self, device, dtype):
        # HermitSymm writes in place, so the caller's tensor has to be copied
        # first while a promoted input is already private.
        x = torch.randn(4, 5, dtype=torch.complex64, device=device).to(dtype)
        before = x.clone()
        expected = torch.ops.aten._fft_c2r.default(x, [1], 0, 8)
        self.assertEqual(x, before, msg=f"functional {dtype}")

        out = torch.empty(expected.shape, dtype=expected.dtype, device=device)
        torch.ops.aten._fft_c2r.out(x, [1], 0, 8, out=out)
        self.assertEqual(x, before, msg=f"out {dtype}")
        self.assertEqual(out, expected, msg=str(dtype))

    @parametrize("op", ("c2c", "r2c", "c2r"))
    def test_fft_out_resizes_output(self, device, op):
        if op == "c2c":
            x = torch.randn(4, 8, dtype=torch.complex128, device=device)
            args, dtype = ([1], 0, True), torch.complex128
        else:
            x = torch.randn(4, 8, dtype=torch.float64, device=device)
            if op == "r2c":
                args, dtype = ([1], 0, True), torch.complex128
            else:
                x = torch.ops.aten._fft_r2c.default(x, [1], 0, True)
                args, dtype = ([1], 0, 8), torch.float64
        fft = getattr(torch.ops.aten, f"_fft_{op}")
        expected = fft.default(x, *args)
        out = torch.empty(0, dtype=dtype, device=device)
        result = fft.out(x, *args, out=out)
        self.assertIs(result, out)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out, expected)

    @parametrize("shape,dims", (((0, 1, 1), (1, 2)), ((0, 4), (1,)), ((2, 0, 4), (2,))))
    def test_fft_out_empty_batch(self, device, shape, dims):
        # An empty batch short-circuits inside _exec_fft, which resizes the
        # destination it was handed. On the direct-write path that destination
        # is the caller's out, so its shape and layout still have to survive.
        msg = f"shape={shape} dims={dims}"
        xc = torch.randn(*shape, dtype=torch.complex128, device=device)
        xr = torch.randn(*shape, dtype=torch.float64, device=device)

        expected = torch.ops.aten._fft_c2c.default(xc, dims, 0, True)
        out = torch.empty(expected.shape, dtype=expected.dtype, device=device)
        stride = out.stride()
        result = torch.ops.aten._fft_c2c.out(xc, dims, 0, True, out=out)
        self.assertIs(result, out)
        self.assertEqual(out.shape, expected.shape, msg=f"c2c {msg}")
        self.assertEqual(out.stride(), stride, f"c2c stride {msg}")

        expected = torch.ops.aten._fft_r2c.default(xr, dims, 0, True)
        out = torch.empty(expected.shape, dtype=expected.dtype, device=device)
        stride = out.stride()
        torch.ops.aten._fft_r2c.out(xr, dims, 0, True, out=out)
        self.assertEqual(out.shape, expected.shape, msg=f"r2c {msg}")
        self.assertEqual(out.stride(), stride, f"r2c stride {msg}")

        last_dim_size = shape[dims[-1]]
        expected = torch.ops.aten._fft_c2r.default(xc, dims, 0, last_dim_size)
        out = torch.empty(expected.shape, dtype=expected.dtype, device=device)
        stride = out.stride()
        torch.ops.aten._fft_c2r.out(xc, dims, 0, last_dim_size, out=out)
        self.assertEqual(out.shape, expected.shape, msg=f"c2r {msg}")
        self.assertEqual(out.stride(), stride, f"c2r stride {msg}")


instantiate_device_type_tests(TestFftOut, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
