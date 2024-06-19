#pragma once
#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void tril_kernel(const Tensor& result, const Tensor& self, int64_t k);

void triu_kernel(const Tensor& result, const Tensor& self, int64_t k);

} // namespace at::native::xpu