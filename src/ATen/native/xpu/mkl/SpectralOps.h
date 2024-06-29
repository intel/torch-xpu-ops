#pragma once

namespace at::native::xpu {

Tensor _fft_c2c_kernel(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward);

} // namespace at::native::xpu
