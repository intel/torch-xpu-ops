#pragma once
// #include <ATen/Dispatch.h>
// #include <ATen/Tensor.h>
// #include <ATen/Utils.h>
// #include <ATen/native/TensorIterator.h>
// #include <ATen/native/sparse/Macros.h>
// #include <ATen/native/SparseTensorUtils.h>

// #ifndef AT_PER_OPERATOR_HEADERS
// #include <ATen/Functions.h>
// #include <ATen/NativeFunctions.h>
// #else
// #include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
// #include <ATen/ops/arange.h>
// #include <ATen/ops/empty.h>
// #include <ATen/ops/tensor.h>
// #endif

#include <ATen/native/sparse/ValidateCompressedIndicesCommon.h>

#define NAME "compressed_index_invariance_checks_xpu"

#define INVARIANT_CHECK_FUNC_API static INLINE FUNCAPI void

namespace at::native {

namespace {

template <typename scalar_t>
using DummyVec = scalar_t;

// template <
//     template <typename func_t>
//     class kernel_t,
//     template <typename func_t, typename vec_func_t>
//     class vec_kernel_t>
// struct KernelLauncher {
//   template <typename func_t, typename vec_func_t>
//   static void launch(
//       TensorIteratorBase& iter,
//       const func_t& f,
//       const vec_func_t& vec_f) {
//     vec_kernel_t<func_t, vec_func_t>::launch(iter, f, vec_f);
//   }

//   template <typename func_t>
//   static void launch(TensorIteratorBase& iter, const func_t& f) {
//     kernel_t<func_t>::launch(iter, f);
//   }
// };

template <CDimName cdim_name, typename index_t>
struct CheckIdxBoundsFunctor {
  index_t operator()(index_t idx) const {
    _check_idx_bounds<cdim_name, index_t>(idx, zero_, dim_);
    return 0;
  }

  CheckIdxBoundsFunctor(index_t zero, index_t dim) : zero_(zero), dim_(dim) {}

 private:
  const index_t zero_;
  const index_t dim_;
};

template <CDimName cdim_name, typename index_t, size_t static_shape_max_len = 0>
struct CheckCIdxFunctor {
  index_t operator()(
      index_t cidx_first,
      index_t cidx_last,
      index_t cidx_curr,
      index_t cidx_next,
      index_t batch_idx) const {
    // Invariant 5.1
    _check_first_cidx_is_zero<cdim_name, index_t>(cidx_first, zero_);
    // Invariant 5.2
    _check_last_cidx_is_nnz<cdim_name, index_t>(cidx_last, nnz_);
    // Invariant 5.3
    _check_cidx_nondecreasing_locally_bounded_sequence<cdim_name, index_t>(
        cidx_curr, cidx_next, zero_, dim_);
    // Invariant 5.6
    // NOTE: the implementation below is sync-less, but,
    // unfortunately, work is not guaranteed to be well-balanced
    // between different threads.
    // Note: 5.6 should not be tested when
    // nnz==0. Fortunately, the code below is no-op when
    // nnz==0.
    int64_t idx_offset = 0;
    // assuming idx contiguity per batch:
    int64_t tmp = batch_idx * nnz_;
    // `nnz == idx_sizes[idx_ndims - 1]` is checked above as `nnz ==
    // idx.size(-1)`
    for (int i = idx_ndims_ - 1; i >= 0 && nnz_ > 0; // break early when nnz==0
         i--) {
      int64_t div = tmp / idx_sizes_[i];
      idx_offset += (tmp - div * idx_sizes_[i]) * idx_strides_[i];
      tmp = div;
    }
    const auto* RESTRICT ptr_idx_batch = ptr_idx_ + idx_offset;
    _check_idx_sorted_distinct_vals_slices_with_cidx<cdim_name, index_t>(
        ptr_idx_batch, cidx_curr, cidx_next);
    return 0;
  }

  CheckCIdxFunctor(
      index_t zero,
      index_t dim,
      int64_t nnz,
      int64_t idx_ndims,
      std::array<int64_t, static_shape_max_len> idx_sizes,
      std::array<int64_t, static_shape_max_len> idx_strides,
      const void* ptr_idx)
      : zero_(zero),
        dim_(dim),
        nnz_(nnz),
        idx_ndims_(idx_ndims),
        idx_sizes_(idx_sizes),
        idx_strides_(idx_strides),
        ptr_idx_(ptr_idx) {}

 private:
  const index_t zero_;
  const index_t dim_;
  int64_t nnz_;
  int64_t idx_ndims_;
  std::array<int64_t, static_shape_max_len> idx_sizes_;
  std::array<int64_t, static_shape_max_len> idx_strides_;
  const void* ptr_idx_;
};

template <
    CDimName cdim_name,
    template <typename func_t>
    class kernel_t,
    template <typename func_t, typename vec_func_t>
    class vec_kernel_t = EmptyVecKernel,
    template <typename scalar_t> class Vec = DummyVec,
    size_t static_shape_max_len = 0>
void _validate_compressed_sparse_indices_kernel_xpu(
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  if (cdim_name == CDimName::CRow) {
    TORCH_CHECK(
        cidx.size(-1) == cdim + 1,
        "crow_indices have wrong shape: ",
        "crow_indices.shape[-1] = ",
        cidx.size(-1),
        " is not equal to ",
        "nrows + 1 = ",
        cdim + 1);
    TORCH_CHECK(
        idx.size(-1) == nnz,
        "col_indices have wrong shape: ",
        "col_indices.shape[-1] = ",
        idx.size(-1),
        " is not equal to ",
        "nnz = ",
        nnz);
  } else {
    TORCH_CHECK(
        cidx.size(-1) == cdim + 1,
        "ccol_indices have wrong shape: ",
        "ccol_indices.shape[-1] = ",
        cidx.size(-1),
        " is not equal to ",
        "ncols + 1 = ",
        cdim + 1);
    TORCH_CHECK(
        idx.size(-1) == nnz,
        "row_indices have wrong shape: ",
        "row_indices.shape[-1] = ",
        idx.size(-1),
        " is not equal to ",
        "nnz = ",
        nnz);
  }

  using KernelLauncher = KernelLauncher<kernel_t, vec_kernel_t>;

  // For TensorIterator's output: no void lambdas.
  const auto dummy = at::empty({1}, cidx.options());

  // Catch integer overflow from large dimensions. Otherwise, the
  // invariant checks may fail with bogus exceptions or succeed with
  // false-positive results when int64_t typed dimensions are cast to
  // index dtype that corresponds to smaller integer type such as
  // int32_t.
  {
    AT_DISPATCH_INDEX_TYPES(idx.scalar_type(), NAME, [cdim, dim, nnz]() {
      if (cdim_name == CDimName::CRow) {
        TORCH_CHECK(
            static_cast<int64_t>(static_cast<index_t>(dim)) == dim,
            sizeof(index_t) * 8,
            "-bit integer overflow in column dimension = ",
            dim);
        TORCH_CHECK(
            static_cast<int64_t>(static_cast<index_t>(cdim)) == cdim,
            sizeof(index_t) * 8,
            "-bit integer overflow in row dimension = ",
            cdim);
      } else {
        TORCH_CHECK(
            static_cast<int64_t>(static_cast<index_t>(dim)) == dim,
            sizeof(index_t) * 8,
            "-bit integer overflow in row dimension = ",
            dim);
        TORCH_CHECK(
            static_cast<int64_t>(static_cast<index_t>(cdim)) == cdim,
            sizeof(index_t) * 8,
            "-bit integer overflow in column dimension = ",
            cdim);
      }
      TORCH_CHECK(
          static_cast<int64_t>(static_cast<index_t>(nnz)) == nnz,
          sizeof(index_t) * 8,
          "-bit integer overflow in nnz = ",
          nnz);
    });
  }

  // Invariants 5.4 and 5.5
  {
    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .add_owned_output(dummy.expand_as(idx))
                    .add_input(idx)
                    .build();

    AT_DISPATCH_INDEX_TYPES(idx.scalar_type(), NAME, [&iter, dim]() {
      const auto zero = index_t{0};
      // change to Functor
      std::cout << "launch CheckIdxBoundsFunctor" << std::endl;
      auto caller = CheckIdxBoundsFunctor<cdim_name, index_t>(zero, dim);
      KernelLauncher::launch(iter, caller);
      // KernelLauncher::launch(iter, [zero, dim] FUNCAPI(index_t idx) ->
      // index_t {
      //   _check_idx_bounds<cdim_name, index_t>(idx, zero, dim);
      //   return 0;
      // });
    });
  }

  // Invariants 5.1, 5.2, 5.3, 5.6
  {
    const auto cidx_first = cidx.slice(-1, 0, 1);
    const auto cidx_last = cidx.slice(-1, cdim, cdim + 1);

    const auto cidx_curr = cidx.slice(-1, 0, cdim);
    const auto cidx_next = cidx.slice(-1, 1, cdim + 1);

    const auto batch_dims = cidx.sizes().slice(0, cidx.dim() - 1);
    const auto batch_count = indexCount(batch_dims);
    const auto batch_idx =
        at::arange(batch_count, cidx.options()).view(batch_dims).unsqueeze_(-1);

    const auto idx_ndims = idx.dim();

    const auto idx_geometry_holder =
        at::sparse::TensorGeometryHolder<static_shape_max_len>(idx);
    const auto idx_sizes = std::get<0>(*idx_geometry_holder);
    const auto idx_strides = std::get<1>(*idx_geometry_holder);

    auto iter = TensorIteratorConfig()
                    .set_check_mem_overlap(false)
                    .add_owned_output(dummy.expand_as(cidx_curr))
                    .add_input(cidx_first)
                    .add_input(cidx_last)
                    .add_input(cidx_curr)
                    .add_input(cidx_next)
                    .add_input(batch_idx)
                    .build();

    AT_DISPATCH_INDEX_TYPES(
        idx.scalar_type(),
        NAME,
        [&iter, &idx, dim, nnz, idx_ndims, &idx_sizes, &idx_strides]() {
          const auto* RESTRICT ptr_idx = idx.const_data_ptr<index_t>();
          const auto zero = index_t{0};
          // KernelLauncher::launch(
          //     iter, CheckCIdxFunctor(zero, dim, nnz, idx_ndims, idx_sizes,
          //     idx_strides, ptr_idx));
          // KernelLauncher::launch(
          //     iter,
          //     [zero, dim, nnz, idx_ndims, idx_sizes, idx_strides, ptr_idx]
          //     FUNCAPI(
          //         index_t cidx_first,
          //         index_t cidx_last,
          //         index_t cidx_curr,
          //         index_t cidx_next,
          //         index_t batch_idx) -> index_t {
          //       // Invariant 5.1
          //       _check_first_cidx_is_zero<cdim_name, index_t>(cidx_first,
          //       zero);
          //       // Invariant 5.2
          //       _check_last_cidx_is_nnz<cdim_name, index_t>(cidx_last, nnz);
          //       // Invariant 5.3
          //       _check_cidx_nondecreasing_locally_bounded_sequence<
          //           cdim_name,
          //           index_t>(cidx_curr, cidx_next, zero, dim);
          //       // Invariant 5.6
          //       // NOTE: the implementation below is sync-less, but,
          //       // unfortunately, work is not guaranteed to be well-balanced
          //       // between different threads.
          //       // Note: 5.6 should not be tested when
          //       // nnz==0. Fortunately, the code below is no-op when
          //       // nnz==0.
          //       int64_t idx_offset = 0;
          //       // assuming idx contiguity per batch:
          //       int64_t tmp = batch_idx * nnz;
          //       // `nnz == idx_sizes[idx_ndims - 1]` is checked above as `nnz
          //       // == idx.size(-1)`
          //       for (int i = idx_ndims - 1;
          //            i >= 0 && nnz > 0;  // break early when nnz==0
          //            i--) {
          //         int64_t div = tmp / idx_sizes[i];
          //         idx_offset += (tmp - div * idx_sizes[i]) * idx_strides[i];
          //         tmp = div;
          //       }
          //       const auto* RESTRICT ptr_idx_batch = ptr_idx + idx_offset;
          //       _check_idx_sorted_distinct_vals_slices_with_cidx<
          //           cdim_name,
          //           index_t>(ptr_idx_batch, cidx_curr, cidx_next);
          //       return 0;
          //     });
        });
  }
}

template <
    template <typename func_t>
    class kernel_t,
    template <typename func_t, typename vec_func_t>
    class vec_kernel_t = EmptyVecKernel,
    template <typename scalar_t> class Vec = DummyVec>
void validate_compressed_sparse_indices_kernel_xpu(
    const bool is_crow,
    const Tensor& cidx,
    const Tensor& idx,
    const int64_t cdim,
    const int64_t dim,
    const int64_t nnz) {
  constexpr size_t idx_max_ndims = 8; // up to 7-dim batch.
  const size_t idx_ndims = static_cast<size_t>(idx.dim());

  if (is_crow) {
    if (idx_ndims <= idx_max_ndims) {
      _validate_compressed_sparse_indices_kernel_xpu<
          CDimName::CRow,
          kernel_t,
          vec_kernel_t,
          Vec,
          idx_max_ndims>(cidx, idx, cdim, dim, nnz);
    } else {
      _validate_compressed_sparse_indices_kernel_xpu<
          CDimName::CRow,
          kernel_t,
          vec_kernel_t,
          Vec>(cidx, idx, cdim, dim, nnz);
    }
  } else {
    if (idx_ndims <= idx_max_ndims) {
      _validate_compressed_sparse_indices_kernel_xpu<
          CDimName::CCol,
          kernel_t,
          vec_kernel_t,
          Vec,
          idx_max_ndims>(cidx, idx, cdim, dim, nnz);
    } else {
      _validate_compressed_sparse_indices_kernel_xpu<
          CDimName::CCol,
          kernel_t,
          vec_kernel_t,
          Vec>(cidx, idx, cdim, dim, nnz);
    }
  }
}

} // namespace

} // namespace at::native
