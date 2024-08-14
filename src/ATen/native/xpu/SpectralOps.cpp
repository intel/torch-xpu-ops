#include <ATen/native/Resize.h>
#include <ATen/native/xpu/mkl/SpectralOps.h>
#include <comm/xpu_aten.h>

namespace at::native {

Tensor _fft_c2c_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_CHECK(self.is_complex());

  return native::xpu::_fft_c2c_mkl(self, dim, normalization, forward);
}

Tensor& _fft_c2c_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());

  return native::xpu::_fft_c2c_mkl_out(self, dim, normalization, forward, out);
}

} // namespace at::native
