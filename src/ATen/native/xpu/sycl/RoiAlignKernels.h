#pragma once

#include <ATen/native/TensorIterator.h>
namespace at::native::xpu {

TORCH_XPU_API Tensor roi_align_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned);

TORCH_XPU_API Tensor roi_align_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned);
} // namespace at::native::xpu