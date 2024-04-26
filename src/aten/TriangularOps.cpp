#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <comm/RegisterUtils.h>

namespace at {

void meta_func_tril(const Tensor& self, int64_t k) {
  TORCH_CHECK(
      self.dim() >= 2, "tril: input tensor must have at least 2 dimensions");
}

Tensor& XPUNativeFunctions::tril_out(
    const Tensor& self,
    int64_t diagonal,
    Tensor& out) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "xpu::tril_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::tril_out", "self");
  meta_func_tril(self, diagonal);
  at::xpu::resize_out(out, self.sizes(), {}, self.options());
  Tensor out_cpu = out.to(Device(kCPU));
  Tensor self_cpu = self.to(Device(kCPU));
  at::tril_out(out_cpu, self_cpu, diagonal);
  out.copy_(out_cpu);
  return out;
}

Tensor XPUNativeFunctions::tril(const Tensor& self, int64_t diagonal) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::tril", "self");
  Tensor out;
  meta_func_tril(self, diagonal);
  out = at::xpu::create_out(self.sizes(), {}, self.options());
  Tensor out_cpu = out.to(Device(kCPU));
  Tensor self_cpu = self.to(Device(kCPU));
  at::tril_out(out_cpu, self_cpu, diagonal);
  out.copy_(out_cpu);
  return out;
}

Tensor& XPUNativeFunctions::tril_(Tensor& self, int64_t diagonal) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::tril", "self");
  meta_func_tril(self, diagonal);
  at::xpu::check_inplace(self, self.sizes(), self.options());
  Tensor self_cpu = self.to(Device(kCPU));
  at::tril_out(self_cpu, self_cpu, diagonal);
  self.copy_(self_cpu);
  return self;
}
} // namespace at