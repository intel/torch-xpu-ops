
#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>

namespace at {

Tensor& XPUNativeFunctions::_fft_r2c_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out) {
  Tensor out_cpu = out.to(Device(at::kCPU));
  at::_fft_r2c_out(
      out_cpu, self.to(Device(at::kCPU)), dim, normalization, onesided);
  return out.copy_(out_cpu);
}

Tensor& XPUNativeFunctions::_fft_c2r_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    Tensor& out) {
  Tensor out_cpu = out.to(Device(at::kCPU));
  Tensor self_cpu = self.to(Device(at::kCPU));
  at::_fft_c2r_out(out_cpu, self_cpu, dim, normalization, last_dim_size);
  out.copy_(out_cpu);
  return out;
}

Tensor& XPUNativeFunctions::_fft_c2c_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  Tensor out_cpu = out.to(Device(at::kCPU));
  Tensor self_cpu = self.to(Device(at::kCPU));
  at::_fft_c2c_out(out_cpu, self_cpu, dim, normalization, forward);
  return out.copy_(out_cpu);
}

Tensor XPUNativeFunctions::_fft_c2r(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size) {
  Tensor out_cpu = at::_fft_c2r(
      self.to(Device(at::kCPU)), dim, normalization, last_dim_size);
  return out_cpu.to(Device(at::kXPU));
}

Tensor XPUNativeFunctions::_fft_r2c(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  Tensor out_cpu =
      at::_fft_r2c(self.to(Device(at::kCPU)), dim, normalization, onesided);
  return out_cpu.to(Device(at::kXPU));
}

Tensor XPUNativeFunctions::_fft_c2c(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  Tensor out_cpu =
      at::_fft_c2c(self.to(Device(at::kCPU)), dim, normalization, forward);
  return out_cpu.to(Device(at::kXPU));
}

} // namespace at
