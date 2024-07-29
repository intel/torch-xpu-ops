#pragma once
#include <ATen/ATen.h>
namespace at::native::xpu {

Tensor repeat_interleave_kernel(
    const Tensor& repeats,
    c10::optional<int64_t> output_size);

} // namespace at::native::xpu