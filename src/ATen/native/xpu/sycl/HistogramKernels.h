#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

void histogramdd_kernel(
    const Tensor& self,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    const Tensor& bin_edges);

void histogramdd_linear_kernel(
    const Tensor& self,
    int64_t bin_ct,
    std::optional<c10::ArrayRef<double>> range,
    const std::optional<Tensor>& weight,
    bool density,
    Tensor& hist,
    Tensor& out_bin_edges);

} // namespace at::native::xpu