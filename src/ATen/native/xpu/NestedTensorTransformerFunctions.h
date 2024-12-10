#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

template <typename T>
TORCH_XPU_API void launch_remove_padding_kernel(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    int batch_size);

template <typename T>
TORCH_XPU_API void launch_remove_padding_transform0213_kernel(
    const T* input,
    T* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int output_dim,
    const int batch_size);

} // namespace at::native::xpu