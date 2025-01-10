#pragma once
#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReductionType.h>

namespace at::native::xpu {

TORCH_XPU_API void index_kernel(
    TensorIteratorBase& iter,
    at::IntArrayRef index_size,
    at::IntArrayRef index_stride);

TORCH_XPU_API void index_select_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& out);

TORCH_XPU_API void masked_fill_kernel(
    TensorIterator& iter,
    const Scalar& value);

TORCH_XPU_API void index_add_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    const Scalar& alpha,
    const Tensor& out);

TORCH_XPU_API void index_fill_kernel(
    TensorIterator& iter,
    const int64_t dim,
    const int64_t self_dim_size,
    const int64_t self_dim_stride,
    const Scalar& source);

TORCH_XPU_API void index_put_kernel(
    TensorIterator& iter,
    IntArrayRef index_size,
    IntArrayRef index_stride,
    bool accumulate);

TORCH_XPU_API void index_put_deterministic_kernel(
    Tensor& self,
    const c10::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    bool accumulate,
    bool unsafe);

TORCH_XPU_API void masked_scatter_kernel(
    const TensorBase& self,
    const TensorBase& mask,
    const TensorBase& maskPrefixSum,
    const TensorBase& source);

TORCH_XPU_API void index_copy_kernel(
    TensorIterator& iter,
    const int64_t dim,
    const int64_t self_dim_size,
    const int64_t self_dim_stride);

TORCH_XPU_API void put_kernel(
    TensorIterator& iter,
    const TensorBase& output,
    const bool accumulate);

TORCH_XPU_API void take_kernel(TensorIterator& iter, const TensorBase& input);

TORCH_XPU_API void index_reduce_prod_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const Tensor& result);

TORCH_XPU_API void index_reduce_mean_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const Tensor& result);

TORCH_XPU_API void index_reduce_amax_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const Tensor& result);

TORCH_XPU_API void index_reduce_amin_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const Tensor& result);

TORCH_XPU_API Tensor index_select_sparse_kernel(
    const Tensor& self,
    int64_t dim,
    const Tensor& index);

} // namespace at::native::xpu
