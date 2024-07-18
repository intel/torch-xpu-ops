#pragma once

namespace at::native::xpu {

Tensor _fft_c2c_kernel(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward);

Tensor _fft_c2r_kernel(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size);

} // namespace at::native::xpu
