#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void random_from_to_kernel(
    TensorIteratorBase& iter,
    uint64_t range,
    int64_t base,
    c10::optional<Generator> gen_);

void random_full_64_bits_range_kernel(
    TensorIteratorBase& iter,
    c10::optional<Generator> gen_);

void random_kernel(TensorIteratorBase& iter, c10::optional<Generator> gen_);

void uniform_kernel(
    TensorIteratorBase& iter,
    double from,
    double to,
    c10::optional<Generator> gen);

void normal_kernel(
    const TensorBase& self,
    double mean,
    double std,
    c10::optional<Generator> gen);

void bernoulli_tensor_kernel(
    const TensorBase& self,
    const TensorBase& p_,
    c10::optional<Generator> gen_);

void bernoulli_scalar_kernel(
    const TensorBase& self,
    double p,
    c10::optional<Generator> gen);

void exponential_kernel(
    TensorIteratorBase& iter,
    double lambda,
    c10::optional<Generator> gen);

} // namespace at::native::xpu
