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
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFftOut(TestCase):
    def test_fft_c2c_out_matches_cpu(self, device):
        real_cpu = torch.randn(2, 3, 4, 5)
        complex_cpu = torch.complex(real_cpu, torch.randn_like(real_cpu))
        complex_xpu = complex_cpu.to(device)

        for dims in ([3], [2, 3], [0], [1], [0, 1], [0, 1, 2], [0, 1, 2, 3]):
            expected = torch.ops.aten._fft_c2c.default(complex_cpu, dims, 0, True)
            out = torch.empty_like(complex_xpu)
            result = torch.ops.aten._fft_c2c.out(complex_xpu, dims, 0, True, out=out)
            self.assertIs(result, out)
            self.assertEqual(out.cpu(), expected)

    def test_fft_out_preserves_contiguous_layout(self, device):
        # Regression: _exec_fft permutes batch/signal dims and re-applies the
        # permuted strides to its destination. Feeding a caller-provided out
        # straight into it corrupts the layout for 55 of the 78 combinations
        # below, so every one of them must keep both its strides and its values.
        for ndim in range(1, 5):
            shape = (2, 3, 4, 5)[-ndim:]
            complex_xpu = torch.randn(*shape, dtype=torch.complex128, device=device)
            real_xpu = torch.randn(*shape, dtype=torch.float64, device=device)
            complex_cpu = complex_xpu.cpu()
            real_cpu = real_xpu.cpu()

            for k in range(1, ndim + 1):
                for dims in itertools.combinations(range(ndim), k):
                    dims = list(dims)
                    msg = f"ndim={ndim} dims={dims}"

                    c2c_ref = torch.ops.aten._fft_c2c.default(
                        complex_xpu, dims, 0, True
                    )
                    # The out= variant is compared against the functional one,
                    # which shares its implementation, so anchor that reference
                    # to CPU to catch an error common to both.
                    self.assertEqual(
                        c2c_ref.cpu(),
                        torch.ops.aten._fft_c2c.default(complex_cpu, dims, 0, True),
                        msg=f"c2c functional {msg}",
                    )
                    # empty_like would inherit the functional result's permuted
                    # strides, so allocate a fresh contiguous buffer instead.
                    out = torch.empty(c2c_ref.shape, dtype=c2c_ref.dtype, device=device)
                    stride = out.stride()
                    torch.ops.aten._fft_c2c.out(complex_xpu, dims, 0, True, out=out)
                    self.assertEqual(out.stride(), stride, f"c2c {msg}")
                    self.assertEqual(out, c2c_ref, msg=f"c2c {msg}")

                    r2c_ref = torch.ops.aten._fft_r2c.default(real_xpu, dims, 0, True)
                    r2c_cpu = torch.ops.aten._fft_r2c.default(real_cpu, dims, 0, True)
                    self.assertEqual(
                        r2c_ref.cpu(), r2c_cpu, msg=f"r2c functional {msg}"
                    )
                    out = torch.empty(r2c_ref.shape, dtype=r2c_ref.dtype, device=device)
                    stride = out.stride()
                    torch.ops.aten._fft_r2c.out(real_xpu, dims, 0, True, out=out)
                    self.assertEqual(out.stride(), stride, f"r2c {msg}")
                    self.assertEqual(out, r2c_ref, msg=f"r2c {msg}")

                    last_dim_size = real_xpu.size(dims[-1])
                    c2r_ref = torch.ops.aten._fft_c2r.default(
                        r2c_ref, dims, 0, last_dim_size
                    )
                    self.assertEqual(
                        c2r_ref.cpu(),
                        torch.ops.aten._fft_c2r.default(
                            r2c_cpu, dims, 0, last_dim_size
                        ),
                        msg=f"c2r functional {msg}",
                    )
                    out = torch.empty(c2r_ref.shape, dtype=c2r_ref.dtype, device=device)
                    stride = out.stride()
                    torch.ops.aten._fft_c2r.out(
                        r2c_ref, dims, 0, last_dim_size, out=out
                    )
                    self.assertEqual(out.stride(), stride, f"c2r {msg}")
                    self.assertEqual(out, c2r_ref, msg=f"c2r {msg}")

    def test_fft_out_noncontiguous_input(self, device):
        # The direct-write predicate inspects the batch strides of the input, so
        # a transposed input has to keep both its values and the out layout.
        x = torch.randn(4, 3, 8, dtype=torch.complex128, device=device).transpose(0, 1)
        for dims in ([2], [1, 2], [0, 1, 2]):
            expected = torch.ops.aten._fft_c2c.default(x, dims, 0, True)
            out = torch.empty(expected.shape, dtype=expected.dtype, device=device)
            stride = out.stride()
            result = torch.ops.aten._fft_c2c.out(x, dims, 0, True, out=out)
            self.assertIs(result, out)
            self.assertEqual(out.stride(), stride, f"dims={dims}")
            self.assertEqual(out, expected, msg=f"dims={dims}")

    def test_fft_r2c_out_preserves_layout_for_size_one_input_strides(self, device):
        # A size-one dimension may have an arbitrary stride while the tensor is
        # still contiguous. The out path must not propagate that input layout to
        # the caller-provided output.
        source = torch.randn(2, 1, 4, device=device)
        x = torch.empty_strided((2, 1, 4), (4, 999, 1), device=device)
        x.copy_(source)
        self.assertTrue(x.is_contiguous())

        for onesided in (True, False):
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

    def test_fft_r2c_out_dtype_promotion(self, device):
        # promote_fft_input promotes half/bfloat16 to float. The functional
        # variant must keep reporting the same output dtype as before the out
        # variants started sharing its implementation.
        for dtype, expected_dtype in (
            (torch.float32, torch.complex64),
            (torch.float64, torch.complex128),
            (torch.half, torch.complex32),
            (torch.bfloat16, torch.complex64),
        ):
            x = torch.randn(8, 16, device=device, dtype=dtype)
            functional = torch.ops.aten._fft_r2c.default(x, [1], 0, True)
            self.assertEqual(functional.dtype, expected_dtype)

            out = torch.empty(functional.shape, dtype=expected_dtype, device=device)
            result = torch.ops.aten._fft_r2c.out(x, [1], 0, True, out=out)
            self.assertIs(result, out)
            self.assertEqual(out.dtype, expected_dtype)
            self.assertEqual(out, functional)

    def test_fft_out_casts_from_the_functional_result_dtype(self, device):
        # The out= contract is functional-result-then-copy: an out tensor with
        # a wider dtype must contain the low-precision functional result cast
        # to that dtype, rather than a result computed directly at the wider
        # transform precision.
        xc = torch.randn(4, 16, dtype=torch.complex64, device=device).to(
            torch.complex32
        )
        xr = torch.randn(4, 16, dtype=torch.float16, device=device)

        for normalization in (0, 1, 2):
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

    def test_fft_half_normalization(self, device):
        # Preserve the established MKL ordering for complex-half c2c/c2r:
        # execute at the promoted precision, cast to the functional result
        # dtype, and normalize that low-precision result in place.
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(2, 3, dtype=torch.complex64, generator=generator).to(
            device=device, dtype=torch.complex32
        )
        c2c_unnormalized = torch.ops.aten._fft_c2c.default(x, [1], 0, True)
        c2r_unnormalized = torch.ops.aten._fft_c2r.default(x, [0, 1], 0, 4)

        for normalization, scale in ((1, 1 / 3**0.5), (2, 1 / 3)):
            got = torch.ops.aten._fft_c2c.default(x, [1], normalization, True)
            self.assertEqual(got.dtype, torch.complex32)
            expected = c2c_unnormalized.clone().mul_(scale)
            self.assertEqual(
                got,
                expected,
                rtol=0,
                atol=0,
                msg=f"c2c normalization={normalization}",
            )

            out = torch.empty(got.shape, dtype=torch.complex32, device=device)
            torch.ops.aten._fft_c2c.out(x, [1], normalization, True, out=out)
            self.assertEqual(out, got, msg=f"normalization={normalization}")

        for normalization, scale in ((1, 1 / 8**0.5), (2, 1 / 8)):
            got = torch.ops.aten._fft_c2r.default(x, [0, 1], normalization, 4)
            self.assertEqual(got.dtype, torch.float16)
            expected = c2r_unnormalized.clone().mul_(scale)
            self.assertEqual(
                got,
                expected,
                rtol=0,
                atol=0,
                msg=f"c2r normalization={normalization}",
            )

            out = torch.empty(got.shape, dtype=torch.float16, device=device)
            torch.ops.aten._fft_c2r.out(x, [0, 1], normalization, 4, out=out)
            self.assertEqual(out, got, msg=f"normalization={normalization}")

    def test_fft_r2c_out_twosided(self, device):
        # onesided=False mirrors the second half in place, the one path that
        # replaces the destination after the transform has already run.
        for ndim, dims in ((1, [0]), (2, [1]), (2, [0, 1]), (3, [2]), (3, [0, 1, 2])):
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

    def test_fft_c2r_does_not_mutate_input(self, device):
        # HermitSymm writes in place, so the caller's tensor has to be copied
        # first while a promoted input is already private.
        for dtype in (torch.complex64, torch.complex32):
            x = torch.randn(4, 5, dtype=torch.complex64, device=device).to(dtype)
            before = x.clone()
            expected = torch.ops.aten._fft_c2r.default(x, [1], 0, 8)
            self.assertEqual(x, before, msg=f"functional {dtype}")

            out = torch.empty(expected.shape, dtype=expected.dtype, device=device)
            torch.ops.aten._fft_c2r.out(x, [1], 0, 8, out=out)
            self.assertEqual(x, before, msg=f"out {dtype}")
            self.assertEqual(out, expected, msg=str(dtype))

    def test_fft_out_resizes_output(self, device):
        complex_xpu = torch.randn(4, 8, dtype=torch.complex128, device=device)
        expected = torch.ops.aten._fft_c2c.default(complex_xpu, [1], 0, True)
        out = torch.empty(0, dtype=torch.complex128, device=device)
        torch.ops.aten._fft_c2c.out(complex_xpu, [1], 0, True, out=out)
        self.assertEqual(out.shape, expected.shape)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out, expected)

        real_xpu = torch.randn(4, 8, dtype=torch.float64, device=device)
        expected = torch.ops.aten._fft_r2c.default(real_xpu, [1], 0, True)
        out = torch.empty(0, dtype=torch.complex128, device=device)
        torch.ops.aten._fft_r2c.out(real_xpu, [1], 0, True, out=out)
        self.assertEqual(out.shape, expected.shape)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out, expected)

        expected = torch.ops.aten._fft_c2r.default(expected, [1], 0, 8)
        out = torch.empty(0, dtype=torch.float64, device=device)
        torch.ops.aten._fft_c2r.out(
            torch.ops.aten._fft_r2c.default(real_xpu, [1], 0, True), [1], 0, 8, out=out
        )
        self.assertEqual(out.shape, expected.shape)
        self.assertTrue(out.is_contiguous())
        self.assertEqual(out, expected)

    def test_fft_out_empty_batch(self, device):
        # An empty batch short-circuits inside _exec_fft, which resizes the
        # destination it was handed. On the direct-write path that destination
        # is the caller's out, so its shape and layout still have to survive.
        for shape, dims in (((0, 1, 1), [1, 2]), ((0, 4), [1]), ((2, 0, 4), [2])):
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
