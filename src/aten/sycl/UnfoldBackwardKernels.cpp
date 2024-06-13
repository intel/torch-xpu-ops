#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/NonEmptyUtils.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <aten/sycl/CopyKernel.h>
#include <aten/sycl/Loops.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

constexpr int n_elems_per_work_item = 4; // UNROLLED_ELEM_PER_WORK_ITEM;

static TensorIterator _make_unfold_backward_iter_over_grad_out(
    Tensor& grad_out,
    const Tensor& grad_in,
    int64_t dim,
    int64_t size,
    int64_t step) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds

  auto grad_out_dim_size = ensure_nonempty_size(grad_out, dim);
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);
  // dictates the number of elements to iterate over in dimension `dim`
  auto iter_dim_size =
      std::min(grad_out_dim_size, (grad_in_dim_size - 1) * step + size);

  /* prepare grad_out for TensorIterator { */
  auto grad_out_strides = ensure_nonempty_vec(grad_out.strides().vec());
  auto grad_out_sizes = ensure_nonempty_vec(grad_out.sizes().vec());
  grad_out_sizes[dim] = iter_dim_size;
  auto grad_out_restrided =
      grad_out.as_strided(grad_out_sizes, grad_out_strides);
  /* } */

  /* prepare grad_in for TensorIterator { */
  auto grad_in_strides = ensure_nonempty_vec(grad_in.strides().vec());
  auto grad_in_sizes = ensure_nonempty_vec(grad_in.sizes().vec());

  // set strides for dim to 0
  // and size to 1 because this dimension is indexed inside the kernel
  grad_in_strides[dim] = 0;
  grad_in_sizes[dim] = 1;

  grad_in_strides.pop_back();
  grad_in_sizes.pop_back();

  auto grad_in_restrided =
      grad_in.squeeze(-1).as_strided(grad_in_sizes, grad_in_strides);
  /* } */

  // During the TensorIterator iteration we have to know
  // i_dim in grad_out[i_1,...,i_dim,...i_n],
  // idx_dim stores this information
  /* prepare idx_dim for TensorIterator { */
  auto idx_dim =
      at::arange(0, iter_dim_size, grad_in.options().dtype(at::kLong));

  auto grad_out_dim = ensure_nonempty_dim(grad_out.dim());

  auto idx_dim_strides = std::vector<int64_t>(grad_out_dim, 0);
  auto idx_dim_sizes = std::vector<int64_t>(grad_out_dim, 1);

  idx_dim_strides[dim] = 1;
  idx_dim_sizes[dim] = iter_dim_size;

  // idx_dim size will broadcast over determined by grad_out sizes in
  // TensorIterator
  auto idx_dim_restrided = idx_dim.as_strided(idx_dim_sizes, idx_dim_strides);
  /* } */

  auto iter = TensorIteratorConfig()
                  .set_check_mem_overlap(false)
                  .check_all_same_dtype(false)
                  .resize_outputs(false)
                  .add_owned_output(grad_out_restrided)
                  .add_owned_const_input(grad_in_restrided)
                  .add_owned_const_input(idx_dim_restrided)
                  .build();

  return iter;
}

template <int n_elems_per_work_item, typename func_t>
struct UnfoldBackwardElementwiseKernelFunctor {
  void operator()(sycl::item<1> item_id) const {
    int idx = item_id.get_linear_id();
#pragma unroll
    for (int i = 0; i < n_elems_per_work_item; ++i) {
      if (idx < total_n_elems_) {
        f_(idx);
        idx += total_work_items_;
      }
    }
  }
  UnfoldBackwardElementwiseKernelFunctor(
      int total_work_items,
      int total_n_elems,
      func_t f)
      : total_work_items_(total_work_items),
        total_n_elems_(total_n_elems),
        f_(f) {}

 private:
  int total_work_items_;
  int total_n_elems_;
  func_t f_;
};

template <int n_elems_per_work_item, typename func_t>
void _unfold_backward_elementwise_kernel(int total_n_elems, func_t f) {
  int total_work_items =
      (total_n_elems + n_elems_per_work_item - 1) / n_elems_per_work_item;
  UnfoldBackwardElementwiseKernelFunctor<n_elems_per_work_item, func_t> kfn(
      total_work_items, total_n_elems, f);
  auto& queue = getCurrentSYCLQueue();
  sycl_kernel_submit(sycl::range<1>(total_work_items), queue, kfn);
}

template <int n_elems_per_work_item, typename func_t>
static void _launch_unfold_backward_kernel(int total_n_elems, func_t f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <=
          std::numeric_limits<int32_t>::max()); // INT_MAX when int32_t

  _unfold_backward_elementwise_kernel<n_elems_per_work_item, func_t>(
      total_n_elems, f);
}

template <typename scalar_t, typename offset_calc_t>
struct UnfoldBackwardFunctor {
  void operator()(int i) const {
    auto offsets = offset_calc_.get(i);

    auto* grad_out_data =
        reinterpret_cast<scalar_t*>(grad_out_ptr_ + offsets[0]);
    auto* grad_in_data = reinterpret_cast<scalar_t*>(grad_in_ptr_ + offsets[1]);

    auto idx_dim = *reinterpret_cast<int64_t*>(idx_dim_ptr_ + offsets[2]);

    // left_fold potentially intersecting with idx_dim
    // is either (idx_dim - size) / step or the next integer.
    int64_t left_fold_idx = (idx_dim > size_) ? (idx_dim - size_) / step_ : 0;
    if (!(left_fold_idx * step_ <= idx_dim &&
          idx_dim < left_fold_idx * step_ + size_)) {
      ++left_fold_idx;
    }

    auto right_fold_idx = idx_dim / step_;
    right_fold_idx = (right_fold_idx >= grad_in_dim_size_)
        ? (grad_in_dim_size_ - 1)
        : right_fold_idx;

    for (auto fold_idx = left_fold_idx; fold_idx <= right_fold_idx;
         ++fold_idx) {
      auto idx_last_dim = idx_dim - fold_idx * step_;
      *grad_out_data += grad_in_data
          [fold_idx * grad_in_dim_stride_ +
           idx_last_dim * grad_in_last_dim_stride_];
    }
  }

  UnfoldBackwardFunctor(
      char* grad_out_ptr,
      char* grad_in_ptr,
      char* idx_dim_ptr,
      offset_calc_t offset_calc,
      int64_t size,
      int64_t step,
      int64_t grad_in_dim_stride,
      int64_t grad_in_last_dim_stride,
      int64_t grad_in_dim_size)
      : grad_out_ptr_(grad_out_ptr),
        grad_in_ptr_(grad_in_ptr),
        idx_dim_ptr_(idx_dim_ptr),
        offset_calc_(offset_calc),
        size_(size),
        step_(step),
        grad_in_dim_stride_(grad_in_dim_stride),
        grad_in_last_dim_stride_(grad_in_last_dim_stride),
        grad_in_dim_size_(grad_in_dim_size) {}

 private:
  char* grad_out_ptr_;
  char* grad_in_ptr_;
  char* idx_dim_ptr_;
  offset_calc_t offset_calc_;
  int64_t size_;
  int64_t step_;
  int64_t grad_in_dim_stride_;
  int64_t grad_in_last_dim_stride_;
  int64_t grad_in_dim_size_;
};

template <typename scalar_t>
void _unfold_backward_internal_kernel(
    TensorIterator& iter,
    int64_t size,
    int64_t step,
    int64_t grad_in_dim_stride,
    int64_t grad_in_last_dim_stride,
    int64_t grad_in_dim_size) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _unfold_backward_internal_kernel<scalar_t>(
          sub_iter,
          size,
          step,
          grad_in_dim_stride,
          grad_in_last_dim_stride,
          grad_in_dim_size);
    }
    return;
  }

  char* grad_out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* grad_in_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* idx_dim_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  auto offset_calc = make_offset_calculator<3>(iter);

  // The algorithm is: for each index in grad_out find
  // the elements contributing to it and sum them up.
  // Note: the algorithm does not require any synchronization.
  UnfoldBackwardFunctor<scalar_t, decltype(offset_calc)> loop(
      grad_out_ptr,
      grad_in_ptr,
      idx_dim_ptr,
      offset_calc,
      size,
      step,
      grad_in_dim_stride,
      grad_in_last_dim_stride,
      grad_in_dim_size);

  _launch_unfold_backward_kernel<n_elems_per_work_item>(iter.numel(), loop);
}

void unfold_backward_kernel(
    Tensor& grad_out,
    const Tensor& grad_in,
    int64_t dim,
    int64_t size,
    int64_t step) {
  dim = maybe_wrap_dim(dim, grad_out.dim());
  // last dim stores the folds
  auto last_dim = maybe_wrap_dim(-1, grad_in.dim());

  auto grad_in_dim_stride = ensure_nonempty_stride(grad_in, dim);
  auto grad_in_last_dim_stride = ensure_nonempty_stride(grad_in, last_dim);
  auto grad_in_dim_size = ensure_nonempty_size(grad_in, dim);

  TensorIterator iter = _make_unfold_backward_iter_over_grad_out(
      grad_out, grad_in, dim, size, step);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "unfold_backward_xpu",
      [&] {
        _unfold_backward_internal_kernel<scalar_t>(
            iter,
            size,
            step,
            grad_in_dim_stride,
            grad_in_last_dim_stride,
            grad_in_dim_size);
      });
}
} // namespace at::native::xpu