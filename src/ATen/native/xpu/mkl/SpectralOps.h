#pragma once

namespace at::native::xpu {

Tensor _fft_c2c_kernel(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward);

Tensor _fft_r2c_kernel(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided);

} // namespace at::native::xpu
