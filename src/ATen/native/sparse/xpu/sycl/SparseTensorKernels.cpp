#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/SparseTensorUtils.h>
#include <c10/macros/Macros.h>
#include <c10/util/accumulate.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_coalesce_native.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/sparse/xpu/sycl/SparseTensorKernels.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

struct CoalesceLTFunctor {
  auto operator()(int64_t a, int64_t b) const {
    return a < b;
  }
};

struct CoalesceEQFunctor {
  template <typename T>
  auto operator()(T lhs, T rhs) const {
    return lhs == rhs;
  }
};

template <typename Dtype, typename Acctype>
struct CoalesceValuesKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    int seg = item.get_group(1) * 4 + item.get_local_id(0);

    // Number of values processed by each thread (grain size)
    constexpr int SZ = 4;

    if (seg < newNnz_) {
      const int newValueRow = seg * stride_;
      const int begin = segment_offsets_[seg];
      const int end = (seg < newNnz_ - 1) ? segment_offsets_[seg + 1] : nnz_;
      const int startFeature = item.get_local_id(1) +
          item.get_group(0) * item.get_local_range(1) * SZ;
      Acctype tmp[SZ];
#pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        tmp[ii] = 0;
      }
      for (int row = begin; row < end; row++) {
        const int valueRow = ((int)value_indices_[row]) * stride_;

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          int featureDim = startFeature + ii * C10_WARP_SIZE;
          if (featureDim < stride_) {
            tmp[ii] += static_cast<Acctype>(values_[valueRow + featureDim]);
          }
        }
      }
#pragma unroll
      for (int ii = 0; ii < SZ; ii++) {
        int featureDim = startFeature + ii * C10_WARP_SIZE;
        if (featureDim < stride_) {
          newValues_[newValueRow + featureDim] = static_cast<Dtype>(tmp[ii]);
        }
      }
    }
  }
  CoalesceValuesKernelFunctor(
      int64_t* segment_offsets,
      int64_t* value_indices,
      Dtype* values,
      Dtype* newValues,
      int64_t nnz,
      int64_t newNnz,
      int64_t stride)
      : segment_offsets_(segment_offsets),
        value_indices_(value_indices),
        values_(values),
        newValues_(newValues),
        nnz_(nnz),
        newNnz_(newNnz),
        stride_(stride) {}

 private:
  int64_t* segment_offsets_;
  int64_t* value_indices_;
  Dtype* values_;
  Dtype* newValues_;
  int64_t nnz_;
  int64_t newNnz_;
  int64_t stride_;
};

SparseTensor coalesce_sparse_kernel(const SparseTensor& self) {
  int64_t nnz = self._nnz();
  TORCH_INTERNAL_ASSERT(!self.is_coalesced());
  // NOTE: Since `coalesce` is not an in-place operation when `is_coalesced` is
  // false, we should keep the original tensor intact and do coalesce on a copy
  // of the tensor
  if (nnz < 2) {
    SparseTensor dst = self.clone();
    dst._coalesced_(true);
    return dst;
  }

  Tensor values = self._values();
  int64_t sparse_dim = self.sparse_dim();

  Tensor indices1D = flatten_indices(self._indices(), self.sizes(), true);
  Tensor origIndices = at::empty({nnz}, self._indices().options());
  Tensor uniqueOffsets = at::empty({nnz}, self._indices().options());

  auto indices_ptr = indices1D.data_ptr<int64_t>();
  auto origIndices_ptr = origIndices.data_ptr<int64_t>();
  auto uniqueOffsets_ptr = uniqueOffsets.data_ptr<int64_t>();

  pstl::iota<int64_t>(origIndices_ptr, origIndices_ptr + nnz, (int64_t)0);
  pstl::iota<int64_t>(uniqueOffsets_ptr, uniqueOffsets_ptr + nnz, (int64_t)0);

  CoalesceLTFunctor lt_functor;
  pstl::sort<int64_t, int64_t>(
      indices_ptr, origIndices_ptr, indices1D.size(0), lt_functor);

  auto indices_end = indices_ptr;
  auto uniqueOffsets_end = uniqueOffsets_ptr;
  CoalesceEQFunctor eq_functor;
  std::tie(indices_end, uniqueOffsets_end) =
      pstl::unique_with_zip<int64_t, int64_t, int64_t>(
          indices_ptr, indices_ptr + nnz, uniqueOffsets_ptr, eq_functor);
  int64_t newNnz = std::distance(indices_ptr, indices_end);

  indices1D.resize_({1, newNnz});
  auto newValues_size = values.sizes().vec();
  newValues_size[0] = newNnz;
  Tensor newValues = at::empty(newValues_size, values.options());

  // If there is no values to copy, save running the kernel.
  if (newValues.numel() > 0) {
    values = values.contiguous();
    int64_t stride = c10::multiply_integers(values.sizes().slice(1));
    sycl::range<2> global_range(
        ceil_div(stride, (int64_t)64 * 4) * 4,
        ceil_div(newNnz, (int64_t)4) * 64);
    sycl::range<2> local_range(4, 64);
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
        at::ScalarType::ComplexHalf,
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        at::ScalarType::Bool,
        values.scalar_type(),
        "coalesce_sparse_xpu",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          auto caller = CoalesceValuesKernelFunctor<scalar_t, accscalar_t>(
              uniqueOffsets.data_ptr<int64_t>(),
              origIndices.data_ptr<int64_t>(),
              values.data_ptr<scalar_t>(),
              newValues.data_ptr<scalar_t>(),
              nnz,
              newNnz,
              stride);
          sycl_kernel_submit(
              global_range, local_range, getCurrentSYCLQueue(), caller);
        });
  }

  // unflatten indices if necessary
  Tensor newIndices;
  if (sparse_dim == 1) {
    newIndices = indices1D;
  } else {
    newIndices = at::empty({sparse_dim, newNnz}, origIndices.options());
    for (int64_t d = sparse_dim - 1; d >= 0; d--) {
      // NB: Not a select, so I can preserve the outer dimension
      Tensor indicesSlice = newIndices.narrow(0, d, 1);
      indicesSlice.copy_(indices1D);
      indices1D.divide_(self.size(d), "trunc");
      indicesSlice.add_(indices1D, -self.size(d));
    }
  }

  // We can use unsafe sparse tensor constructor because the indices do not
  // need to be revalidated as we do not add or change indices, just remove
  // duplicates.
  SparseTensor dst = ::at::native::_sparse_coo_tensor_unsafe(
                         newIndices, newValues, self.sizes())
                         ._coalesced_(true);
  return dst;
}

} // namespace at::native::xpu
