
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/TriangularOpsKernels.h>
#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/xpu/ops/tril_native.h>
#include <ATen/xpu/ops/triu_native.h>

namespace at::native {

TORCH_IMPL_FUNC(tril_xpu)(const Tensor& self, int64_t k, const Tensor& result) {
  if (self.numel() != 0) {
    xpu::tril_kernel(result, self, k);
  }
}

TORCH_IMPL_FUNC(triu_xpu)(const Tensor& self, int64_t k, const Tensor& result) {
  if (self.numel() != 0) {
    xpu::triu_kernel(result, self, k);
  }
}
// void tril_meta(const Tensor& self, int64_t k) {
//   TORCH_CHECK(
//       self.dim() >= 2, "tril: input tensor must have at least 2 dimensions");
// }

// Tensor& XPUNativeFunctions::tril_out(
//     const Tensor& self,
//     int64_t diagonal,
//     Tensor& out) {
//   std::optional<Device> common_device = std::nullopt;
//   c10::impl::check_and_update_common_device(
//       common_device, out, "xpu::tril_out", "out");
//   c10::impl::check_and_update_common_device(
//       common_device, self, "xpu::tril_out", "self");
//   tril_meta(self, diagonal);
//   xpu::resize_out(out, self.sizes(), {}, self.options());
//   return native::xpu::tril_kernel(out, self, diagonal);
// }

// Tensor XPUNativeFunctions::tril(const Tensor& self, int64_t diagonal) {
//   tril_meta(self, diagonal);
//   Tensor out = xpu::create_out(self.sizes(), {}, self.options());
//   return tril_out(self, diagonal, out);
// }

// Tensor& XPUNativeFunctions::tril_(Tensor& self, int64_t diagonal) {
//   tril_meta(self, diagonal);
//   xpu::check_inplace(self, self.sizes(), self.options());
//   return tril_out(self, diagonal, self);
// }

// void triu_meta(const Tensor& self, int64_t k) {
//   TORCH_CHECK(
//       self.dim() >= 2, "triu: input tensor must have at least 2 dimensions");
// }

// Tensor& XPUNativeFunctions::triu_out(
//     const Tensor& self,
//     int64_t diagonal,
//     Tensor& out) {
//   std::optional<Device> common_device = std::nullopt;
//   c10::impl::check_and_update_common_device(
//       common_device, out, "xpu::triu_out", "out");
//   c10::impl::check_and_update_common_device(
//       common_device, self, "xpu::triu_out", "self");
//   triu_meta(self, diagonal);
//   xpu::resize_out(out, self.sizes(), {}, self.options());
//   return native::xpu::triu_kernel(out, self, diagonal);
// }

// Tensor XPUNativeFunctions::triu(const Tensor& self, int64_t diagonal) {
//   triu_meta(self, diagonal);
//   Tensor out = xpu::create_out(self.sizes(), {}, self.options());
//   return triu_out(self, diagonal, out);
// }

// Tensor& XPUNativeFunctions::triu_(Tensor& self, int64_t diagonal) {
//   triu_meta(self, diagonal);
//   xpu::check_inplace(self, self.sizes(), self.options());
//   return triu_out(self, diagonal, self);
// }
} // namespace at::native
