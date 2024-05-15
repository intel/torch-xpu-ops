#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

Tensor& tril_kernel(Tensor& result, const Tensor& self, int64_t k);

Tensor& triu_kernel(Tensor& result, const Tensor& self, int64_t k);

} // namespace at::native::xpu