#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void uniform_kernel(
    TensorIterator& iter,
    double from,
    double to,
    c10::optional<Generator> gen);

void normal_kernel(
    TensorIterator& iter,
    double mean,
    double std,
    c10::optional<Generator> gen);

void bernoulli_compare_kernel(TensorIterator& iter);

void bernoulli_float_compare_kernel(
    TensorIterator& iter,
    double p,
    c10::optional<Generator> gen);

} // namespace at::native::xpu
