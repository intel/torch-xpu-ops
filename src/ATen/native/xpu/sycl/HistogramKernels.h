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

TORCH_XPU_API void histogram_select_outer_bin_edges_kernel(
    const Tensor& input,
    const int64_t N,
    std::vector<double>& leftmost_edges,
    std::vector<double>& rightmost_edges);

} // namespace at::native::xpu
