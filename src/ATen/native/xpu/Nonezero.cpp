#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/nonzero.h>
#endif
#include <ATen/xpu/XPUNativeFunctions.h>

namespace at {

at::Tensor XPUNativeFunctions::nonzero(const at::Tensor& self) {
  // TODO: SYCL implementation
  return at::nonzero(self.to("cpu")).to("xpu");
}

} // namespace at
