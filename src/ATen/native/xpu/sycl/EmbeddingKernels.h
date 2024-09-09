#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

Tensor embedding_dense_backward_kernel(
    const Tensor& grad_,
    const Tensor& indices_,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq);

Tensor& embedding_renorm_kernel(
    Tensor& self,
    const Tensor& indices,
    double max_norm,
    double norm_type);

} // namespace at::native::xpu
