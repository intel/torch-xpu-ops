#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <aten/sycl/Indexing.h>

namespace at {

Tensor& XPUNativeFunctions::index_select_out(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  return at::native::xpu::index_select_out_kernel(self, dim, index, out);
}

Tensor XPUNativeFunctions::index_select(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  auto out = at::empty({0}, self.options());
  return at::native::xpu::index_select_out_kernel(self, dim, index, out);
}
} // namespace at