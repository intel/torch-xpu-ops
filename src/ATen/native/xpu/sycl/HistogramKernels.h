#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API void histogramdd_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges_);

TORCH_XPU_API void histogramdd_linear_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const TensorList& bin_edges_,
    bool local_search);

} // namespace at::native::xpu
