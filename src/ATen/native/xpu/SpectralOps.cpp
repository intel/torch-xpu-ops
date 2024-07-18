#include <ATen/ATen.h>
#include <ATen/native/xpu/mkl/SpectralOps.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

Tensor XPUNativeFunctions::_fft_c2c(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_CHECK(self.is_complex());

  if (dim.empty()) {
    return self.clone();
  }

  return native::xpu::_fft_c2c_kernel(self, dim, normalization, forward);
}

Tensor& XPUNativeFunctions::_fft_c2c_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());

  if (dim.empty()) {
    out.copy_(self);
    return out;
  }

  Tensor result =
      native::xpu::_fft_c2c_kernel(self, dim, normalization, forward);
  out.copy_(result);
  return out;
}

Tensor XPUNativeFunctions::_fft_c2r(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size) {
  TORCH_CHECK(self.is_complex());

  if (dim.empty()) {
    return self.clone();
  }

  return native::xpu::_fft_c2r_kernel(self, dim, normalization, last_dim_size);
}

} // namespace at
