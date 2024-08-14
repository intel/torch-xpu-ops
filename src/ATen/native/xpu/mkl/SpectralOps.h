#pragma once

namespace at::native::xpu {

Tensor _fft_c2c_mkl(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward);

Tensor& _fft_c2c_mkl_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out);

} // namespace at::native::xpu
