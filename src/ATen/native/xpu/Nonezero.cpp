
#include <ATen/ops/nonzero.h>
#include <xpu/ATen/ops/nonzero_native.h>

namespace at::native {

at::Tensor nonzero_xpu(const at::Tensor& self) {
  // TODO: SYCL implementation
  return at::nonzero(self.to("cpu")).to("xpu");
}

at::Tensor& nonzero_out_xpu(const at::Tensor& self, Tensor& out) {
  // TODO: SYCL implementation
  return out = nonzero_xpu(self);
}

} // namespace at::native
