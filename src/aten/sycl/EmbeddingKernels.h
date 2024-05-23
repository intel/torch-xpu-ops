#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

std::tuple<Tensor, Tensor, Tensor, Tensor> _embedding_bag_dpcpp(
    const Tensor& weight_t,
    const Tensor& indices_t,
    const Tensor& offsets_t,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const Tensor& per_sample_weights_t,
    bool include_last_offset,
    int64_t padding_idx);

Tensor embedding_dense_backward_kernel(
    const Tensor& grad_,
    const Tensor& indices_,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq);

} // namespace xpu
} // namespace native
} // namespace at
