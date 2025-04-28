#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

TORCH_XPU_API Tensor _fft_c2c_mkl(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward);

TORCH_XPU_API Tensor& _fft_c2c_mkl_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out);

TORCH_XPU_API Tensor _fft_c2r_mkl(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size);

TORCH_XPU_API Tensor& _fft_c2r_mkl_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    Tensor& out);

TORCH_XPU_API Tensor _fft_r2c_mkl(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided);

TORCH_XPU_API Tensor& _fft_r2c_mkl_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out);

} // namespace at::native::xpu
