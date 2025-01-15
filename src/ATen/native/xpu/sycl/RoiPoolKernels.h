#pragma once

#include <ATen/native/TensorIterator.h>
namespace at::native::xpu {

TORCH_XPU_API std::tuple<Tensor, Tensor> roi_pool_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width);

TORCH_XPU_API Tensor roi_pool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width);
} // namespace at::native::xpu