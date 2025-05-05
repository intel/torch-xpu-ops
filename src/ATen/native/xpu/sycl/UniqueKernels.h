#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> unique_consecutive_kernel(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    std::optional<int64_t> dim);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> unique_dim_consecutive_kernel(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse,
    const bool return_counts);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> unique_dim_kernel(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse,
    const bool return_counts);

TORCH_XPU_API std::tuple<Tensor, Tensor> _unique_kernel(
    const Tensor& self,
    const bool return_inverse);

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor> _unique2_kernel(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts);

} // namespace at::native::xpu
