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

#if defined(__MKL_FALLBACK_TO_CPU)
  Tensor out_cpu = native::_fft_c2c_mkl(
      self.to(Device(at::kCPU)), dim, normalization, forward);
  return out_cpu.to(Device(at::kXPU));
#else
  return native::xpu::_fft_c2c_mkl(self, dim, normalization, forward);
#endif // __MKL_FALLBACK_TO_CPU
}

Tensor& XPUNativeFunctions::_fft_c2c_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());

#if defined(__MKL_FALLBACK_TO_CPU)
  Tensor out_cpu = out.to(Device(at::kCPU));
  native::_fft_c2c_mkl_out(
      self.to(Device(at::kCPU)), dim, normalization, forward, out_cpu);
  out.copy_(out_cpu);
  return out;
#else
  return native::xpu::_fft_c2c_mkl_out(self, dim, normalization, forward, out);
#endif // __MKL_FALLBACK_TO_CPU
}

} // namespace at
