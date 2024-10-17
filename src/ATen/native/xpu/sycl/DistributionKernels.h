#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void random_from_to_kernel(
    TensorIteratorBase& iter,
    uint64_t range,
    int64_t base,
    c10::optional<Generator> gen_);

TORCH_XPU_API void random_full_64_bits_range_kernel(
    TensorIteratorBase& iter,
    c10::optional<Generator> gen_);

TORCH_XPU_API void random_kernel(
    TensorIteratorBase& iter,
    c10::optional<Generator> gen_);

TORCH_XPU_API void uniform_kernel(
    TensorIteratorBase& iter,
    double from,
    double to,
    c10::optional<Generator> gen);

TORCH_XPU_API void normal_kernel(
    const TensorBase& self,
    double mean,
    double std,
    c10::optional<Generator> gen);

TORCH_XPU_API void bernoulli_tensor_kernel(
    const TensorBase& self,
    const TensorBase& p_,
    c10::optional<Generator> gen_);

TORCH_XPU_API void bernoulli_scalar_kernel(
    const TensorBase& self,
    double p,
    c10::optional<Generator> gen);

TORCH_XPU_API void exponential_kernel(
    TensorIteratorBase& iter,
    double lambda,
    c10::optional<Generator> gen);

TORCH_XPU_API void cauchy_kernel(
    TensorIteratorBase& iter,
    double median,
    double sigma,
    c10::optional<Generator> gen);

TORCH_XPU_API void log_normal_kernel(
    TensorIteratorBase& iter,
    double mean,
    double std,
    std::optional<Generator> gen);

} // namespace at::native::xpu
