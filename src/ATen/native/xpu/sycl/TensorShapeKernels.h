#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void split_with_sizes_copy_out_xpu_kernel(
    const Tensor& self,
    IntArrayRef split_sizes,
    int64_t dim,
    TensorList out);

TORCH_XPU_API Tensor
_chunk_cat_xpu_kernel(TensorList tensors, int64_t dim, int64_t num_chunks);

TORCH_XPU_API Tensor& _chunk_cat_out_xpu_kernel(
    TensorList tensors,
    int64_t dim,
    int64_t num_chunks,
    Tensor& out);

} // namespace at::native::xpu
