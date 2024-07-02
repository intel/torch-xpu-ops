#include <ATen/xpu/XPUNativeFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/is_set_to_native.h>
#endif

namespace at {

bool XPUNativeFunctions::is_set_to(const Tensor& self, const Tensor& src) {
  return at::native::is_set_to(self, src);
}

} // namespace at
