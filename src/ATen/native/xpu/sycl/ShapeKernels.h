#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void cat_out_kernel(
    const ITensorListRef& tensors,
    int64_t dim,
    int64_t valid,
    bool all_contiguous,
    bool all_same_dtype,
    bool all_same_sizes_and_stride,
    MemoryFormat memory_format,
    const Tensor& result);

} // namespace at::native::xpu
