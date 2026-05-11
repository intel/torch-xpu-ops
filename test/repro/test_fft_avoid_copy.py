# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Reproducer for: Consider how to avoid copy in FFT kernels
# Verifies that _fft_c2c_mkl_out, _fft_c2r_mkl_out, and _fft_r2c_mkl_out
# produce correct results when writing directly into a pre-allocated output
# tensor (i.e. when the `out=` argument is supplied), without an unnecessary
# intermediate copy_.

import pytest
import torch


requires_xpu = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="XPU device not available"
)

DTYPES_C2C = [torch.complex64, torch.complex128]
DTYPES_R2C = [torch.float32, torch.float64]


# ---------------------------------------------------------------------------
# _fft_c2c_mkl_out  (torch.fft.fft / torch.fft.ifft with out= argument)
# ---------------------------------------------------------------------------


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_C2C)
@pytest.mark.parametrize("n", [63, 64])  # odd and even lengths
def test_fft_c2c_out_1d(dtype, n):
    x = torch.randn(n, dtype=dtype, device="xpu")
    out = torch.empty_like(x)
    result = torch.fft.fft(x, out=out)
    expected = torch.fft.fft(x)
    assert result.data_ptr() == out.data_ptr(), "result must share storage with out"
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_C2C)
def test_ifft_c2c_out_1d(dtype):
    x = torch.randn(64, dtype=dtype, device="xpu")
    out = torch.empty_like(x)
    result = torch.fft.ifft(x, out=out)
    expected = torch.fft.ifft(x)
    assert result.data_ptr() == out.data_ptr(), "result must share storage with out"
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_C2C)
def test_fft_c2c_out_2d(dtype):
    x = torch.randn(8, 16, dtype=dtype, device="xpu")
    out = torch.empty_like(x)
    result = torch.fft.fft2(x, out=out)
    expected = torch.fft.fft2(x)
    assert result.data_ptr() == out.data_ptr(), "result must share storage with out"
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_C2C)
def test_fft_c2c_out_norm_modes(dtype):
    norms = [None, "forward", "backward", "ortho"]
    x = torch.randn(32, dtype=dtype, device="xpu")
    for norm in norms:
        out = torch.empty_like(x)
        result = torch.fft.fft(x, norm=norm, out=out)
        expected = torch.fft.fft(x, norm=norm)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)


# ---------------------------------------------------------------------------
# _fft_r2c_mkl_out  (torch.fft.rfft with out= argument)
# ---------------------------------------------------------------------------


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_R2C)
@pytest.mark.parametrize("n", [63, 64])
def test_fft_r2c_out_1d(dtype, n):
    x = torch.randn(n, dtype=dtype, device="xpu")
    out = torch.empty(n // 2 + 1, dtype=torch.complex64 if dtype == torch.float32 else torch.complex128, device="xpu")
    result = torch.fft.rfft(x, out=out)
    expected = torch.fft.rfft(x)
    assert result.data_ptr() == out.data_ptr(), "result must share storage with out"
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_R2C)
def test_fft_r2c_out_2d(dtype):
    x = torch.randn(8, 16, dtype=dtype, device="xpu")
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    out = torch.empty(8, 9, dtype=cdtype, device="xpu")
    result = torch.fft.rfft2(x, out=out)
    expected = torch.fft.rfft2(x)
    assert result.data_ptr() == out.data_ptr(), "result must share storage with out"
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_R2C)
def test_fft_r2c_out_twosided(dtype):
    n = 64
    x = torch.randn(n, dtype=dtype, device="xpu")
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    out = torch.empty(n, dtype=cdtype, device="xpu")
    result = torch.fft.fft(x.to(cdtype), out=out)
    expected = torch.fft.fft(x.to(cdtype))
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)


# ---------------------------------------------------------------------------
# _fft_c2r_mkl_out  (torch.fft.irfft with out= argument)
# ---------------------------------------------------------------------------


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_R2C)
@pytest.mark.parametrize("n", [62, 64])
def test_fft_c2r_out_1d(dtype, n):
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    x = torch.randn(n // 2 + 1, dtype=cdtype, device="xpu")
    out = torch.empty(n, dtype=dtype, device="xpu")
    result = torch.fft.irfft(x, n=n, out=out)
    expected = torch.fft.irfft(x, n=n)
    assert result.data_ptr() == out.data_ptr(), "result must share storage with out"
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_R2C)
def test_fft_c2r_out_2d(dtype):
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    x = torch.randn(8, 9, dtype=cdtype, device="xpu")
    out = torch.empty(8, 16, dtype=dtype, device="xpu")
    result = torch.fft.irfft2(x, s=(8, 16), out=out)
    expected = torch.fft.irfft2(x, s=(8, 16))
    assert result.data_ptr() == out.data_ptr(), "result must share storage with out"
    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)


@requires_xpu
@pytest.mark.parametrize("dtype", DTYPES_R2C)
def test_fft_c2r_out_norm_modes(dtype):
    norms = [None, "forward", "backward", "ortho"]
    n = 64
    cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    x = torch.randn(n // 2 + 1, dtype=cdtype, device="xpu")
    for norm in norms:
        out = torch.empty(n, dtype=dtype, device="xpu")
        result = torch.fft.irfft(x, n=n, norm=norm, out=out)
        expected = torch.fft.irfft(x, n=n, norm=norm)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-5)
