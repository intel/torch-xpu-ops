#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void fractional_max_pool3d_kernel(
    const Tensor& input,
    int64_t poolSizeT,
    int64_t poolSizeH,
    int64_t poolSizeW,
    int64_t outputT,
    int64_t outputH,
    int64_t outputW,
    const Tensor& randomSamples,
    int64_t numBatch,
    int64_t numPlanes,
    int64_t inputT,
    int64_t inputH,
    int64_t inputW,
    const Tensor& output,
    const Tensor& indices);

TORCH_XPU_API void fractional_max_pool3d_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef output_size,
    const Tensor& indices);

} // namespace at::native::xpu
