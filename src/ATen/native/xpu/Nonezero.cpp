// #ifndef AT_PER_OPERATOR_HEADERS
// #include <ATen/Functions.h>
// #include <ATen/NativeFunctions.h>
// #else
// #include <ATen/ops/nonzero.h>
// #endif

#include <ATen/ops/nonzero.h>
#include <ATen/xpu/ops/nonzero_native.h>

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
