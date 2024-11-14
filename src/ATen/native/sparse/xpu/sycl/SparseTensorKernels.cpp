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

#include <ATen/native/sparse/FlattenIndicesCommon.h>
#include <ATen/native/sparse/xpu/sycl/SparseTensorKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>
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

template <template <typename func_t> class kernel_t>
struct KernelLauncher {
  template <typename func_t>
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    kernel_t<func_t>::launch(iter, f);
  }
};

template <typename index_t, typename hash_coeffs_t>
struct FlattenIndicesFunctor {
  int64_t operator()(int64_t nnz_idx) const {
    const auto* ptr_indices_dim = ptr_indices_ + nnz_idx * indices_nnz_stride_;
    auto hash = static_cast<int64_t>(0);
    for (int64_t dim = 0; dim < sparse_dim_; ++dim) {
      const auto dim_hash_coeff = hash_coeffs_[dim];
      const auto dim_index = ptr_indices_dim[dim * indices_dim_stride_];
      hash += dim_index * dim_hash_coeff;
    }
    return hash;
  }
  FlattenIndicesFunctor(
      const index_t* ptr_indices,
      int64_t indices_nnz_stride,
      int64_t sparse_dim,
      hash_coeffs_t hash_coeffs,
      int64_t indices_dim_stride)
      : ptr_indices_(ptr_indices),
        indices_nnz_stride_(indices_nnz_stride),
        sparse_dim_(sparse_dim),
        hash_coeffs_(hash_coeffs),
        indices_dim_stride_(indices_dim_stride) {}

 private:
  const index_t* ptr_indices_;
  int64_t indices_nnz_stride_;
  int64_t sparse_dim_;
  hash_coeffs_t hash_coeffs_;
  int64_t indices_dim_stride_;
};

template <
    template <typename func_t>
    class kernel_t,
    typename index_t,
    int64_t max_static_len = 0>
Tensor _flatten_indices_impl(const Tensor& indices, IntArrayRef size) {
  TORCH_INTERNAL_ASSERT(
      indices.dim() > 1 && static_cast<size_t>(indices.size(0)) == size.size());

  // Need owning storage in case of the Tensor class.
  const auto hash_coeffs_storage = [&]() -> auto{
    auto strides = c10::contiguous_strides(size);
    return at::sparse::TensorGeometryHolder<max_static_len>(
        strides, strides, indices.options());
  }
  ();
  const auto hash_coeffs = std::get<0>(*hash_coeffs_storage);

  const auto hash_indices = [&]() -> Tensor {
    // non-const because of gcc-5/clang-5 issues
    auto sparse_dim = indices.size(0);
    auto indices_dim_stride = indices.stride(0);
    auto indices_nnz_stride = indices.stride(1);

    auto hash = at::arange(indices.size(1), indices.options().dtype(kLong));

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .add_output(hash)
                    .add_input(hash)
                    .build();

    {
      const auto* ptr_indices = indices.const_data_ptr<index_t>();
      KernelLauncher<kernel_t>::launch(
          iter,
          FlattenIndicesFunctor<index_t, decltype(hash_coeffs)>(
              ptr_indices,
              indices_nnz_stride,
              sparse_dim,
              hash_coeffs,
              indices_dim_stride));
    }

    return hash;
  }();

  return hash_indices;
}

template <template <typename func_t> class kernel_t>
Tensor _flatten_indices(const Tensor& indices, IntArrayRef size) {
  TORCH_CHECK(
      indices.dim() > 1 && static_cast<size_t>(indices.size(0)) == size.size(),
      NAME,
      "(): the dimensionality of sparse `indices` and the length of `size` must match. ",
      "Got `indices.size(0) == ",
      indices.size(0),
      "` != `size.size() == ",
      size.size(),
      "`.");
  Tensor flattened_indices;
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), NAME, [&]() {
    constexpr int64_t max_sparse_dims = 8;
    if (indices.size(0) <= max_sparse_dims) {
      flattened_indices =
          _flatten_indices_impl<kernel_t, index_t, max_sparse_dims>(
              indices, size);
    } else {
      flattened_indices =
          _flatten_indices_impl<kernel_t, index_t>(indices, size);
    }
  });
  return flattened_indices;
}

template <typename func_t>
struct FlattenIndicesKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    gpu_kernel(iter, f);
  }
};

Tensor flatten_indices_kernel(const Tensor& indices, IntArrayRef size) {
  return _flatten_indices<FlattenIndicesKernelLauncher>(indices, size);
}

} // namespace at::native::xpu
