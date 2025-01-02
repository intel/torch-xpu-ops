#pragma once

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

} // namespace at::native::xpu
