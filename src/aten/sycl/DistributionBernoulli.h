#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void bernoulli_compare_kernel(TensorIterator& iter);

void bernoulli_compare_scalar_kernel(
    TensorIterator& iter,
    double p,
    c10::optional<Generator> gen);

} // namespace at::native::xpu
