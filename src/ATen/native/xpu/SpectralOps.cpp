#include <ATen/ATen.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/mkl/SpectralOps.h>
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

Tensor XPUNativeFunctions::_fft_c2c(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_CHECK(self.is_complex());

  return native::xpu::_fft_c2c_mkl(self, dim, normalization, forward);
}

Tensor& XPUNativeFunctions::_fft_c2c_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());

  return native::xpu::_fft_c2c_mkl_out(self, dim, normalization, forward, out);
}

} // namespace at
