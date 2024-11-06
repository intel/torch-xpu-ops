#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/FunctionOfAMatrixUtilsKernels.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>

#include <comm/SYCLContext.h>

constexpr int n_elems_per_work_item = 4; // UNROLLED_ELEM_PER_WORK_ITEM;

namespace at::native::xpu {

template <int n_elems_per_work_item, typename func_t>
struct _ElemwiseKernelFunctor {
  void operator()(sycl::item<1> itemId) const {
    int idx = itemId.get_linear_id();
#pragma unroll
    for (int i = 0; i < n_elems_per_work_item; ++i) {
      if (idx < total_n_elems_) {
        f_(idx);
        idx += total_work_items_;
      }
    }
  }
  _ElemwiseKernelFunctor(int total_n_elems, func_t f, int total_work_items)
      : total_n_elems_(total_n_elems),
        f_(f),
        total_work_items_(total_work_items) {}

 private:
  int total_n_elems_;
  func_t f_;
  int total_work_items_;
};

template <int n_elems_per_work_item, typename func_t>
void _elemwise_kernel(int total_n_elems, func_t f) {
  int total_work_items =
      (total_n_elems + n_elems_per_work_item - 1) / n_elems_per_work_item;
  _ElemwiseKernelFunctor<n_elems_per_work_item, func_t> kfn(
      total_n_elems, f, total_work_items);
  sycl_kernel_submit(
      sycl::range<1>(total_work_items), getCurrentSYCLQueue(), kfn);
}

template <int n_elems_per_work_item, typename func_t>
void _lauch_kernel(int total_n_elems, const func_t& f) {
  TORCH_INTERNAL_ASSERT(
      total_n_elems >= 0 &&
      total_n_elems <= std::numeric_limits<int32_t>::max());

  _elemwise_kernel<n_elems_per_work_item, func_t>(total_n_elems, f);
}

template <typename scalar_t, typename offset_calc_t>
struct ComputeLinearCombinationInternalKernelFunctor {
  void operator()(int idx) const {
    auto offsets = offset_calc_.get(idx);

    auto* RESTRICT out_data =
        reinterpret_cast<scalar_t*>(out_ptr_ + offsets[0]);
    auto* RESTRICT in_data = reinterpret_cast<scalar_t*>(in_ptr_ + offsets[1]);
    using primitive_t = typename scalar_value_type<scalar_t>::type;
    auto* RESTRICT coeff_data =
        reinterpret_cast<primitive_t*>(coeff_ptr_ + offsets[2]);

    // perform summation
    for (int32_t i = 0; i < num_summations_; ++i) {
      *out_data += in_data[i * in_stride_] * coeff_data[i * coeff_stride_];
    }
  }

  ComputeLinearCombinationInternalKernelFunctor(
      offset_calc_t offset_calc,
      char* RESTRICT out_ptr,
      char* RESTRICT in_ptr,
      char* RESTRICT coeff_ptr,
      int32_t num_summations,
      int32_t in_stride,
      int32_t coeff_stride)
      : offset_calc_(offset_calc),
        out_ptr_(out_ptr),
        in_ptr_(in_ptr),
        coeff_ptr_(coeff_ptr),
        num_summations_(num_summations),
        in_stride_(in_stride),
        coeff_stride_(coeff_stride) {}

 private:
  offset_calc_t offset_calc_;
  char* RESTRICT out_ptr_;
  char* RESTRICT in_ptr_;
  char* RESTRICT coeff_ptr_;
  int32_t num_summations_;
  int32_t in_stride_;
  int32_t coeff_stride_;
};

template <typename scalar_t>
void _compute_linear_combination_internal_kernel(
    TensorIterator& iter,
    int32_t in_stride,
    int32_t coeff_stride,
    int32_t num_summations) {
  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      _compute_linear_combination_internal_kernel<scalar_t>(
          sub_iter, in_stride, coeff_stride, num_summations);
    }
    return;
  }

  auto offset_calc = make_offset_calculator<3>(iter);
  char* RESTRICT out_ptr = reinterpret_cast<char*>(iter.data_ptr(0));
  char* RESTRICT in_ptr = reinterpret_cast<char*>(iter.data_ptr(1));
  char* RESTRICT coeff_ptr = reinterpret_cast<char*>(iter.data_ptr(2));

  ComputeLinearCombinationInternalKernelFunctor<scalar_t, decltype(offset_calc)>
      loop(
          offset_calc,
          out_ptr,
          in_ptr,
          coeff_ptr,
          num_summations,
          in_stride,
          coeff_stride);

  _lauch_kernel<n_elems_per_work_item>(iter.numel(), loop);
}

void _compute_linear_combination_kernel(
    TensorIterator& iter,
    int64_t in_stride,
    int64_t coeff_stride,
    int64_t num_summations) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::Bool,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "_compute_linear_combination_xpu",
      [&]() {
        _compute_linear_combination_internal_kernel<scalar_t>(
            iter, in_stride, coeff_stride, num_summations);
      });
}
} // namespace at::native::xpu
