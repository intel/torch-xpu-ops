#pragma once

#include <ATen/ATen.h>

namespace at::native::xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_kernel(
    const Tensor& weight_t,
    const Tensor& indices_t,
    const Tensor& offsets_t,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights_t,
    bool include_last_offset,
    int64_t padding_idx);

} // namespace at::native::xpu
