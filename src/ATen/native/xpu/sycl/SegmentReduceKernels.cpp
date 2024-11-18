#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/core/Scalar.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReductionType.h>
#include <optional>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_segment_reduce_backward_native.h>
#include <ATen/ops/all.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/segment_reduce_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <ATen/native/xpu/sycl/SegmentReduceKernels.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t, typename index_t>
struct SegmentReduceForwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t idx = item.get_global_linear_id();
    auto initial_value = initial_value_raw_;
    if (idx >= size_) {
      return;
    }
    int64_t row_id = idx / inner_offset_;
    int64_t lane_id = idx % inner_offset_; // lane_id is the inner_idx
    int64_t outer_idx = row_id / segment_count_;
    int64_t dim_idx = row_id % segment_count_;

    int64_t offset_idx =
        outer_idx * lengths_cumsum_stride_axis_ * (segment_count_ + 1) +
        dim_idx;
    index_t offset_start = lengths_cumsum_data_[offset_idx];
    index_t offset_end = lengths_cumsum_data_[offset_idx + 1];

    // ===== step2: apply reduction_
    for (index_t j = offset_start; j < offset_end; ++j) {
      int64_t data_index = outer_idx * data_stride_axis_ * data_size_axis_ +
          j * data_stride_axis_ + lane_id;
      const auto data = values_data_[data_index];
      // TODO: There is no need to branch with every element
      if (reduction_ == native::ReductionType::MAX) {
        initial_value =
            std::isnan(data) ? data : std::max<scalar_t>(initial_value, data);
      } else if (
          reduction_ == native::ReductionType::MEAN ||
          reduction_ == native::ReductionType::SUM) {
        initial_value = initial_value + data;
      } else if (reduction_ == native::ReductionType::MIN) {
        initial_value =
            std::isnan(data) ? data : std::min<scalar_t>(initial_value, data);
      } else if (reduction_ == native::ReductionType::PROD) {
        initial_value = initial_value * data;
      }
    }

    // ===== step3: finalize reduction_
    int64_t lengths_idx =
        outer_idx * lengths_stride_axis_ * segment_count_ + dim_idx;
    SYCL_KERNEL_ASSERT(lengths_data_[lengths_idx] >= 0);
    if (lengths_data_[lengths_idx] == 0 && !is_initial_set_ &&
        reduction_ == native::ReductionType::MEAN) {
      initial_value = static_cast<scalar_t>(NAN);
    } else if (
        reduction_ == native::ReductionType::MEAN &&
        lengths_data_[lengths_idx] > 0 && !std::isnan(initial_value)) {
      initial_value = initial_value / lengths_data_[lengths_idx];
    }
    int64_t output_index = outer_idx * output_stride_axis_ * output_size_axis_ +
        dim_idx * output_stride_axis_ + lane_id;
    output_data_[output_index] = initial_value;
  }

  SegmentReduceForwardKernelFunctor(
      native::ReductionType reduction,
      scalar_t* output_data,
      const scalar_t* values_data,
      const index_t* lengths_data,
      const index_t* lengths_cumsum_data,
      const int64_t segment_count,
      const int64_t lengths_stride_axis,
      bool is_initial_set,
      scalar_t initial_value_raw,
      const int64_t outer_offset,
      const int64_t inner_offset,
      const int64_t data_stride_axis,
      const int64_t data_size_axis,
      const int64_t output_stride_axis,
      const int64_t output_size_axis,
      const int64_t lengths_cumsum_stride_axis,
      const int64_t size)
      : reduction_(reduction),
        output_data_(output_data),
        values_data_(values_data),
        lengths_data_(lengths_data),
        lengths_cumsum_data_(lengths_cumsum_data),
        segment_count_(segment_count),
        lengths_stride_axis_(lengths_stride_axis),
        is_initial_set_(is_initial_set),
        initial_value_raw_(initial_value_raw),
        outer_offset_(outer_offset),
        inner_offset_(inner_offset),
        data_stride_axis_(data_stride_axis),
        data_size_axis_(data_size_axis),
        output_stride_axis_(output_stride_axis),
        output_size_axis_(output_size_axis),
        lengths_cumsum_stride_axis_(lengths_cumsum_stride_axis),
        size_(size) {}

 private:
  native::ReductionType reduction_;
  scalar_t* output_data_;
  const scalar_t* values_data_;
  const index_t* lengths_data_;
  const index_t* lengths_cumsum_data_;
  const int64_t segment_count_;
  const int64_t lengths_stride_axis_;
  bool is_initial_set_;
  scalar_t initial_value_raw_;
  const int64_t outer_offset_;
  const int64_t inner_offset_;
  const int64_t data_stride_axis_;
  const int64_t data_size_axis_;
  const int64_t output_stride_axis_;
  const int64_t output_size_axis_;
  const int64_t lengths_cumsum_stride_axis_;
  const int64_t size_;
};

template <typename scalar_t, typename index_t>
void segment_reduce_forward_kernel(
    native::ReductionType reduction,
    scalar_t* output_data,
    const scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t lengths_stride_axis,
    bool is_initial_set,
    scalar_t initial_value_raw,
    const int64_t outer_offset,
    const int64_t inner_offset,
    const int64_t data_stride_axis,
    const int64_t data_size_axis,
    const int64_t output_stride_axis,
    const int64_t output_size_axis,
    const int64_t lengths_cumsum_stride_axis) {
  const int64_t size = outer_offset * segment_count * inner_offset;
  using Kernel = SegmentReduceForwardKernelFunctor<scalar_t, index_t>;
  const int64_t work_group_size = syclMaxWorkGroupSize<Kernel>();
  const int64_t work_group_num = (size + work_group_size - 1) / work_group_size;
  Kernel kfn(
      reduction,
      output_data,
      values_data,
      lengths_data,
      lengths_cumsum_data,
      segment_count,
      lengths_stride_axis,
      is_initial_set,
      initial_value_raw,
      outer_offset,
      inner_offset,
      data_stride_axis,
      data_size_axis,
      output_stride_axis,
      output_size_axis,
      lengths_cumsum_stride_axis,
      size);

  sycl_kernel_submit(
      work_group_size * work_group_num,
      work_group_size,
      getCurrentSYCLQueue(),
      kfn);
}

Tensor _segment_reduce_lengths_offsets_xpu_kernel(
    native::ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths_or_offsets,
    int64_t axis,
    const std::optional<Scalar>& initial,
    bool is_offsets_like) {
  // data and lengths_or_offsets should be contiguous from the call to
  // .contiguous in segment_reduce_kernel
  TORCH_CHECK(data.is_contiguous());
  TORCH_CHECK(lengths_or_offsets.is_contiguous());
  axis = lengths_or_offsets.dim() - 1;
  int64_t segment_count = is_offsets_like ? lengths_or_offsets.size(axis) - 1
                                          : lengths_or_offsets.size(axis);
  int64_t lengths_stride_axis = lengths_or_offsets.stride(axis);
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  auto offsets = lengths_or_offsets;
  auto lengths = lengths_or_offsets;
  if (is_offsets_like) {
    lengths = lengths.diff();
  } else {
    // _get_complete_sum only supports 1D
    auto zeros_shape = offsets.sizes().vec();
    zeros_shape[axis] = 1;
    offsets =
        at::cat({at::zeros(zeros_shape, offsets.options()), offsets}, axis);
    offsets.cumsum_(axis);
  }

  // outer_offset is the size of the outer dimensions of output (before axis)
  // inner_offset is the size of the inner dimensions of output (after axis)
  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++) {
    outer_offset *= output.size(d);
  }
  for (int64_t d = axis + 1; d < output.dim(); d++) {
    inner_offset *= output.size(d);
  }

  constexpr int threads_per_block = 256;
  // segment_count * stride_count is just output.numel() ?
  int64_t num_blocks =
      (output.numel() + threads_per_block - 1) / threads_per_block;

  num_blocks = std::max(num_blocks, (int64_t)1);

  auto data_stride_axis = data.stride(axis);
  auto data_size_axis = data.size(axis);
  auto output_stride_axis = output.stride(axis);
  auto output_size_axis = output.size(axis);
  auto offsets_stride_axis = offsets.stride(axis);

  AT_DISPATCH_INDEX_TYPES(
      lengths_or_offsets.scalar_type(), "_segment_reduce_xpu_kernel1", ([&] {
        auto* offsets_data_ptr = offsets.const_data_ptr<index_t>();
        auto* lengths_data_ptr = lengths.const_data_ptr<index_t>();
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            data.scalar_type(),
            "segment_reduce_xpu",
            [&]() {
              auto* data_data_ptr = data.const_data_ptr<scalar_t>();
              auto* output_data_ptr = output.mutable_data_ptr<scalar_t>();

              // initialize starting value
              scalar_t initial_value;
              if (initial.has_value()) {
                initial_value = initial.value().to<scalar_t>();
              } else if (reduction == native::ReductionType::MAX) {
                initial_value = -std::numeric_limits<scalar_t>::infinity();
              } else if (
                  reduction == native::ReductionType::MEAN ||
                  reduction == native::ReductionType::SUM) {
                initial_value = 0;
              } else if (reduction == native::ReductionType::MIN) {
                initial_value = std::numeric_limits<scalar_t>::infinity();
              } else if (reduction == native::ReductionType::PROD) {
                initial_value = 1;
              }
              segment_reduce_forward_kernel<scalar_t>(
                  reduction,
                  output_data_ptr,
                  data_data_ptr,
                  lengths_data_ptr,
                  offsets_data_ptr,
                  segment_count,
                  lengths_stride_axis,
                  initial.has_value(),
                  initial_value,
                  outer_offset,
                  inner_offset,
                  data_stride_axis,
                  data_size_axis,
                  output_stride_axis,
                  output_size_axis,
                  offsets_stride_axis);
            });
      }));

  return output;
}

Tensor _segment_reduce_lengths_kernel(
    native::ReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_xpu_kernel(
      reduction, data, lengths, axis, initial, /*is_offsets_like=*/false);
}

Tensor _segment_reduce_offsets_kernel(
    native::ReductionType reduction,
    const Tensor& data,
    const Tensor& offsets,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_xpu_kernel(
      reduction, data, offsets, axis, initial, /*is_offsets_like=*/true);
}

template <typename scalar_t, typename index_t>
struct SegmentReduceBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t idx = item.get_global_linear_id();
    if (idx >= size_) {
      return;
    }
    if (idx >= size_) {
      return;
    }
    int64_t row_id = idx / inner_offset_;
    int64_t lane_id = idx % inner_offset_; // lane_id is the inner_idx
    int64_t outer_idx = row_id / segment_count_;
    int64_t dim_idx = row_id % segment_count_;

    int64_t lengths_idx =
        outer_idx * lengths_stride_axis_ * segment_count_ + dim_idx;
    auto segment_length = lengths_data_[lengths_idx];
    if (segment_length == 0) {
      return;
    }

    int64_t offset_idx =
        outer_idx * lengths_cumsum_stride_axis_ * (segment_count_ + 1) +
        dim_idx;
    index_t offset_start = lengths_cumsum_data_[offset_idx];
    index_t offset_end = lengths_cumsum_data_[offset_idx + 1];

    int64_t output_index = outer_idx * output_stride_axis_ * output_size_axis_ +
        dim_idx * output_stride_axis_ + lane_id;

    if (reduction_ == native::ReductionType::MAX ||
        reduction_ == native::ReductionType::MIN) {
      int64_t counter = 0;
      for (int64_t j = offset_start; j < offset_end; ++j) {
        int64_t data_index = outer_idx * data_stride_axis_ * data_size_axis_ +
            j * data_stride_axis_ + lane_id;
        if (std::isnan(values_data_[data_index]) ||
            values_data_[data_index] == output_data_[output_index]) {
          grad_input_data_[data_index] = grad_data_[output_index];
          counter++;
        }
      }
      // Average gradient based on number of maximum elements in the
      // segment
      if (counter < 2) {
        return;
      }
      for (int64_t j = offset_start; j < offset_end; ++j) {
        int64_t data_index = outer_idx * data_stride_axis_ * data_size_axis_ +
            j * data_stride_axis_ + lane_id;
        if (grad_input_data_[data_index] > 0) {
          grad_input_data_[data_index] = grad_input_data_[data_index] / counter;
        }
      }
    } else if (reduction_ == native::ReductionType::MEAN) {
      auto grad_val = grad_data_[output_index] / segment_length;
      for (int64_t j = offset_start; j < offset_end; ++j) {
        int64_t data_index = outer_idx * data_stride_axis_ * data_size_axis_ +
            j * data_stride_axis_ + lane_id;
        grad_input_data_[data_index] = grad_val;
      }
    } else if (reduction_ == native::ReductionType::SUM) {
      const auto& grad_val = grad_data_[output_index];
      for (int64_t j = offset_start; j < offset_end; ++j) {
        int64_t data_index = outer_idx * data_stride_axis_ * data_size_axis_ +
            j * data_stride_axis_ + lane_id;
        grad_input_data_[data_index] = grad_val;
      }
    } else if (reduction_ == native::ReductionType::PROD) {
      const auto& grad_val =
          grad_data_[output_index] * output_data_[output_index];
      for (int64_t j = offset_start; j < offset_end; ++j) {
        int64_t data_index = outer_idx * data_stride_axis_ * data_size_axis_ +
            j * data_stride_axis_ + lane_id;
        if (std::isnan(values_data_[data_index]) ||
            values_data_[data_index] == 0) {
          // explicitly compute exclusive prod
          scalar_t exclusive_prod = initial_prod_value_;
          int64_t prod_idx;
          for (int64_t k = offset_start; k < offset_end; ++k) {
            if (k != j) {
              prod_idx = outer_idx * data_stride_axis_ * data_size_axis_ +
                  k * data_stride_axis_ + lane_id;
              exclusive_prod *= values_data_[prod_idx];
            }
          }
          grad_input_data_[data_index] =
              grad_data_[output_index] * exclusive_prod;
        } else {
          grad_input_data_[data_index] = grad_val / values_data_[data_index];
        }
      }
    }
  }

  SegmentReduceBackwardKernelFunctor(
      native::ReductionType reduction,
      scalar_t* grad_input_data,
      const scalar_t* grad_data,
      const scalar_t* output_data,
      const scalar_t* values_data,
      const index_t* lengths_data,
      const index_t* lengths_cumsum_data,
      const int64_t segment_count,
      const int64_t lengths_stride_axis,
      scalar_t initial_prod_value,
      const int64_t outer_offset,
      const int64_t inner_offset,
      const int64_t data_stride_axis,
      const int64_t data_size_axis,
      const int64_t output_stride_axis,
      const int64_t output_size_axis,
      const int64_t lengths_cumsum_stride_axis,
      const int64_t size)
      : reduction_(reduction),
        grad_input_data_(grad_input_data),
        grad_data_(grad_data),
        output_data_(output_data),
        values_data_(values_data),
        lengths_data_(lengths_data),
        lengths_cumsum_data_(lengths_cumsum_data),
        segment_count_(segment_count),
        lengths_stride_axis_(lengths_stride_axis),
        initial_prod_value_(initial_prod_value),
        outer_offset_(outer_offset),
        inner_offset_(inner_offset),
        data_stride_axis_(data_stride_axis),
        data_size_axis_(data_size_axis),
        output_stride_axis_(output_stride_axis),
        output_size_axis_(output_size_axis),
        lengths_cumsum_stride_axis_(lengths_cumsum_stride_axis),
        size_(size) {}

 private:
  native::ReductionType reduction_;
  scalar_t* grad_input_data_;
  const scalar_t* grad_data_;
  const scalar_t* output_data_;
  const scalar_t* values_data_;
  const index_t* lengths_data_;
  const index_t* lengths_cumsum_data_;
  const int64_t segment_count_;
  const int64_t lengths_stride_axis_;
  scalar_t initial_prod_value_;
  const int64_t outer_offset_;
  const int64_t inner_offset_;
  const int64_t data_stride_axis_;
  const int64_t data_size_axis_;
  const int64_t output_stride_axis_;
  const int64_t output_size_axis_;
  const int64_t lengths_cumsum_stride_axis_;
  const int64_t size_;
};

template <typename scalar_t, typename index_t>
void segment_reduce_backward_kernel(
    native::ReductionType reduction,
    scalar_t* grad_input_data,
    const scalar_t* grad_data,
    const scalar_t* output_data,
    const scalar_t* values_data,
    const index_t* lengths_data,
    const index_t* lengths_cumsum_data,
    const int64_t segment_count,
    const int64_t lengths_stride_axis,
    scalar_t initial_prod_value,
    const int64_t outer_offset,
    const int64_t inner_offset,
    const int64_t data_stride_axis,
    const int64_t data_size_axis,
    const int64_t output_stride_axis,
    const int64_t output_size_axis,
    const int64_t lengths_cumsum_stride_axis) {
  const int64_t size = outer_offset * segment_count * inner_offset;
  using Kernel = SegmentReduceBackwardKernelFunctor<scalar_t, index_t>;
  const int64_t work_group_size = syclMaxWorkGroupSize<Kernel>();
  const int64_t work_group_num = (size + work_group_size - 1) / work_group_size;

  Kernel kfn(
      reduction,
      grad_input_data,
      grad_data,
      output_data,
      values_data,
      lengths_data,
      lengths_cumsum_data,
      segment_count,
      lengths_stride_axis,
      initial_prod_value,
      outer_offset,
      inner_offset,
      data_stride_axis,
      data_size_axis,
      output_stride_axis,
      output_size_axis,
      lengths_cumsum_stride_axis,
      size);

  sycl_kernel_submit(
      work_group_size * work_group_num,
      work_group_size,
      getCurrentSYCLQueue(),
      kfn);
}

Tensor _segment_reduce_lengths_offsets_backward_xpu_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    native::ReductionType reduction,
    const Tensor& lengths_or_offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial,
    bool is_offsets_like) {
  axis = lengths_or_offsets_contig.dim() - 1;
  int64_t segment_count = is_offsets_like
      ? lengths_or_offsets_contig.size(axis) - 1
      : lengths_or_offsets_contig.size(axis);
  int64_t lengths_stride_axis = lengths_or_offsets_contig.stride(axis);
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  auto offsets = lengths_or_offsets_contig;
  auto lengths = lengths_or_offsets_contig;
  if (is_offsets_like) {
    lengths = lengths.diff();
  } else {
    // _get_complete_sum only supports 1D
    auto zeros_shape = offsets.sizes().vec();
    zeros_shape[axis] = 1;
    offsets =
        at::cat({at::zeros(zeros_shape, offsets.options()), offsets}, axis);
    offsets.cumsum_(axis);
  }

  // outer_offset is the size of the outer dimensions of output (before axis)
  // inner_offset is the size of the inner dimensions of output (after axis)
  int64_t outer_offset = 1, inner_offset = 1;
  for (int64_t d = 0; d < axis; d++) {
    outer_offset *= output_contig.size(d);
  }
  for (int64_t d = axis + 1; d < output_contig.dim(); d++) {
    inner_offset *= output_contig.size(d);
  }

  constexpr int threads_per_block = 256;
  int64_t num_blocks =
      (outer_offset * inner_offset * segment_count + threads_per_block - 1) /
      threads_per_block;

  num_blocks = std::max(num_blocks, (int64_t)1);

  auto data_stride_axis = data_contig.stride(axis);
  auto data_size_axis = data_contig.size(axis);
  auto output_stride_axis = output_contig.stride(axis);
  auto output_size_axis = output_contig.size(axis);
  auto offsets_stride_axis = offsets.stride(axis);

  AT_DISPATCH_INDEX_TYPES(
      lengths_or_offsets_contig.scalar_type(),
      "_segment_reduce_xpu_lengths_offsets_backward_kernel1",
      ([&] {
        const auto* lengths_data = lengths.const_data_ptr<index_t>();
        auto* offsets_data = offsets.const_data_ptr<index_t>();

        // TODO: Switch to TensorIterator for better maintainablility and
        // readability
        AT_DISPATCH_FLOATING_TYPES_AND2(
            kBFloat16,
            kHalf,
            data_contig.scalar_type(),
            "_segment_reduce_xpu",
            ([&]() {
              auto* output_data = output_contig.const_data_ptr<scalar_t>();
              auto* grad_data = grad_contig.const_data_ptr<scalar_t>();
              auto* grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
              const auto* values_data = data_contig.const_data_ptr<scalar_t>();

              scalar_t initial_prod_value;
              if (initial.has_value()) {
                initial_prod_value = initial.value().to<scalar_t>();
              } else {
                initial_prod_value = 1;
              }

              segment_reduce_backward_kernel<scalar_t>(
                  reduction,
                  grad_input_data,
                  grad_data,
                  output_data,
                  values_data,
                  lengths_data,
                  offsets_data,
                  segment_count,
                  lengths_stride_axis,
                  initial_prod_value,
                  outer_offset,
                  inner_offset,
                  data_stride_axis,
                  data_size_axis,
                  output_stride_axis,
                  output_size_axis,
                  offsets_stride_axis);
            }));
      }));
  return grad_input;
}

Tensor _segment_reduce_lengths_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    native::ReductionType reduction,
    const Tensor& lengths_contig,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_backward_xpu_kernel(
      grad_contig,
      output_contig,
      data_contig,
      reduction,
      lengths_contig,
      axis,
      initial,
      /*is_offsets_like=*/false);
}

Tensor _segment_reduce_offsets_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    native::ReductionType reduction,
    const Tensor& offsets_contig,
    int64_t axis,
    const std::optional<Scalar>& initial) {
  return _segment_reduce_lengths_offsets_backward_xpu_kernel(
      grad_contig,
      output_contig,
      data_contig,
      reduction,
      offsets_contig,
      axis,
      initial,
      /*is_offsets_like=*/true);
}

} // namespace at::native::xpu
