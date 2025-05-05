#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor> dropout_kernel(
    const Tensor& self,
    double p,
    std::optional<bool> train);

TORCH_XPU_API Tensor
dropout_backward_kernel(const Tensor& grad, const Tensor& mask, double scale);

TORCH_XPU_API std::tuple<Tensor, Tensor> fused_dropout_kernel(
    const Tensor& self,
    double p,
    std::optional<Generator> gen_);

TORCH_XPU_API Tensor
masked_scale_kernel(const Tensor& self, const Tensor& mask, double scale);

} // namespace xpu
} // namespace native
} // namespace at
