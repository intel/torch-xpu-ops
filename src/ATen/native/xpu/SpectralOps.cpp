#if defined(USE_ONEMKL)
#include <ATen/native/xpu/mkl/SpectralOps.h>
#else
#include <ATen/native/Resize.h>
#include <ATen/ops/_fft_c2c_native.h>
#endif // USE_ONEMKL

namespace at::native {

Tensor _fft_c2c_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_CHECK(self.is_complex());

#if defined(USE_ONEMKL)
  return native::xpu::_fft_c2c_mkl(self, dim, normalization, forward);
#else
  Tensor out_cpu = native::_fft_c2c_mkl(
      self.to(Device(at::kCPU)), dim, normalization, forward);
  return out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL
}

Tensor& _fft_c2c_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());

#if defined(USE_ONEMKL)
  return native::xpu::_fft_c2c_mkl_out(self, dim, normalization, forward, out);
#else
  Tensor out_cpu = native::_fft_c2c_mkl(
      self.to(Device(at::kCPU)), dim, normalization, forward);
  at::native::resize_output(out, out_cpu.sizes());
  out.copy_(out_cpu);
  return out;
#endif // USE_ONEMKL
}

} // namespace at::native
