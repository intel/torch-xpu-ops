#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void sqrt_kernel(TensorIteratorBase& iter);

void rsqrt_kernel(TensorIteratorBase& iter);

void bitwise_not_kernel(TensorIteratorBase& iter);

void exp_kernel(TensorIteratorBase& iter);

void expm1_kernel(TensorIteratorBase& iter);

void nan_to_num_kernel(
    TensorIteratorBase& iter,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf);

} // namespace at::native::xpu
