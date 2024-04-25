#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
// #include <c10/core/SymIntArrayRef.h>
#include <comm/RegisterUtils.h>
// #include <torch/library.h>

namespace at {

void torch_meta_func_tril(const Tensor& self, int64_t k, Tensor& out) {
  TORCH_CHECK(
      self.dim() >= 2, "tril: input tensor must have at least 2 dimensions")
  at::xpu::resize_out(out, self.sizes(), {}, self.options());
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
  torch_meta_func_tril(self, diagonal, out);
  Tensor out_cpu = out.to(Device(kCPU));
  Tensor self_cpu = self.to(Device(kCPU));
  at::tril_out(out_cpu, self_cpu, diagonal);
  out.copy_(out_cpu);
  return out;
}
} // namespace at