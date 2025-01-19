#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void remove_padding_kernel_float(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

TORCH_XPU_API void remove_padding_kernel_half(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

TORCH_XPU_API void remove_padding_transform0213_kernel_float(
    const float* input,
    float* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

TORCH_XPU_API void remove_padding_transform0213_kernel_half(
    const c10::Half* input,
    c10::Half* output,
    const int* offsets,
    const int* input_sizes,
    const int* output_sizes,
    int64_t output_dim,
    const int64_t batch_size);

TORCH_XPU_API void add_padding_kernel(
    at::Tensor input,
    at::Tensor output,
    double padding,
    const at::Tensor offsets,
    const at::Tensor nt_sizes,
    int input_dim,
    const std::vector<int64_t>& new_size,
    const int batch_size,
    const int output_batch_size);

} // namespace at::native::xpu
