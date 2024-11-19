#pragma once

#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/pstl/PSTLFunctions.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

using namespace at::xpu;
constexpr int64_t NROWS_PER_THREAD = 64;

template <typename T, typename V>
inline auto CeilDiv(T a, V b) {
  return (a + b - 1) / b;
}

template <typename index_t>
struct KrnPartialsPerSegmentKernelFunctor {
  void operator()(sycl::item<1> item) const {
    auto ret_ptr = ret_data_;
    auto offsets_ptr = offsets_data_;
    auto id = item.get_linear_id();
    if (id < num_of_segments_) {
      const index_t idx_start = offsets_ptr[id];
      const index_t idx_end = (id == num_of_segments_ - 1)
          ? static_cast<index_t>(numel_)
          : offsets_ptr[id + 1];
      const index_t size = idx_end - idx_start;
      ret_ptr[id] = CeilDiv(size, static_cast<index_t>(NROWS_PER_THREAD));
    }
  }
  KrnPartialsPerSegmentKernelFunctor(
      index_t* ret_data,
      const index_t* offsets_data,
      index_t num_of_segments,
      int64_t numel)
      : ret_data_(ret_data),
        offsets_data_(offsets_data),
        num_of_segments_(num_of_segments),
        numel_(numel) {}

 private:
  index_t* ret_data_;
  const index_t* offsets_data_;
  index_t num_of_segments_;
  int64_t numel_;
};

template <typename index_t>
void krn_partials_per_segment(
    index_t* ret,
    const index_t* segment_offsets,
    index_t num_of_segments,
    int64_t numel) {
  auto caller = KrnPartialsPerSegmentKernelFunctor<index_t>(
      ret, segment_offsets, num_of_segments, numel);
  auto range = sycl::range<1>((size_t)num_of_segments);
  sycl_kernel_submit(range, getCurrentSYCLQueue(), caller);
}

template <typename index_t>
struct KrnPartialSegmentOffsetKernelFunctor {
  void operator()(sycl::item<1> item) const {
    auto ret_ptr = ret_data_;
    auto partials_per_segment_ptr = partials_per_segment_data_;
    auto partials_per_segment_offset_ptr = partials_per_segment_offset_data_;
    auto segment_offsets_ptr = segment_offsets_data_;

    auto id = item.get_linear_id();
    if (id < num_of_segments_) {
      index_t idx = partials_per_segment_offset_ptr[id];
      const index_t num_partials = partials_per_segment_ptr[id];
      const index_t segment_offset = segment_offsets_ptr[id];
      for (index_t i = 0; i < num_partials; ++i) {
        ret_ptr[idx++] = segment_offset + i * NROWS_PER_THREAD;
      }
    }
  }
  KrnPartialSegmentOffsetKernelFunctor(
      index_t* ret_data,
      const index_t* partials_per_segment_data,
      const index_t* partials_per_segment_offset_data,
      const index_t* segment_offsets_data,
      index_t num_of_segments)
      : ret_data_(ret_data),
        partials_per_segment_data_(partials_per_segment_data),
        partials_per_segment_offset_data_(partials_per_segment_offset_data),
        segment_offsets_data_(segment_offsets_data),
        num_of_segments_(num_of_segments) {}

 private:
  index_t* ret_data_;
  const index_t* partials_per_segment_data_;
  const index_t* partials_per_segment_offset_data_;
  const index_t* segment_offsets_data_;
  index_t num_of_segments_;
};

template <typename index_t>
void krn_partial_segment_offset(
    index_t* ret,
    const index_t* partials_per_segment,
    const index_t* partials_per_segment_offset,
    const index_t* segment_offsets,
    index_t num_of_segments) {
  auto caller = KrnPartialSegmentOffsetKernelFunctor<index_t>(
      ret,
      partials_per_segment,
      partials_per_segment_offset,
      segment_offsets,
      num_of_segments);
  auto range = sycl::range<1>((size_t)num_of_segments);
  sycl_kernel_submit(range, getCurrentSYCLQueue(), caller);
}

template <typename scalar_t, typename index_t>
struct ComputeGradWeightBagsKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto grad_weight_per_segment_ptr = grad_weight_per_segment_data_;
    auto indices_ptr = indices_data_;
    auto gradoutput_ptr = gradoutput_data_;
    auto offset2bag_ptr = offset2bag_data_;
    auto count_ptr = count_defined_ ? count_data_ : NULL;
    auto bag_size_ptr = bag_size_data_;
    auto per_sample_weights_ptr =
        per_sample_weight_defined_ ? per_sample_weights_data_ : NULL;
    auto segment_offsets_ptr = segment_offsets_data_;

    const int gid = item.get_global_linear_id();
    const int id = gid / stride_warped_;
    const int startFeature = gid % stride_warped_;
    if (startFeature >= stride_) {
      return;
    }
    if (id >= num_of_segments_) {
      return;
    }

    const int idx_begin = segment_offsets_ptr[id];
    const int idx_end =
        (id == num_of_segments_ - 1) ? numel_ : segment_offsets_ptr[id + 1];

    acc_type_device<scalar_t, kXPU> weight = 0.f;
    for (int idx = idx_begin; idx < idx_end; ++idx) {
      const int orig_row = indices_ptr[idx];
      const int seq_number = offset2bag_ptr[orig_row];
      const int grad_output_row = seq_number * stride_;

      acc_type_device<scalar_t, kXPU> scale =
          count_ptr ? 1.f / count_ptr[idx] : 1.f;
      if (per_sample_weight_defined_) {
        scale *= per_sample_weights_ptr[orig_row * per_sample_weights_stride_];
      }

      acc_type_device<scalar_t, kXPU> gradient =
          gradoutput_ptr[grad_output_row + startFeature];
      if (mode_mean_) {
        gradient /= bag_size_ptr[seq_number];
      }
      weight += gradient * scale;
    }
    grad_weight_per_segment_ptr[id * stride_ + startFeature] = weight;
  }
  ComputeGradWeightBagsKernelFunctor(
      int64_t numel,
      int64_t stride,
      int mode_mean,
      int64_t num_of_segments,
      int64_t stride_warped,
      bool per_sample_weight_defined,
      bool count_defined,
      int64_t per_sample_weights_stride,
      acc_type_device<scalar_t, kXPU>* grad_weight_per_segment_data,
      index_t* indices_data,
      const scalar_t* gradoutput_data,
      index_t* offset2bag_data,
      index_t* count_data,
      index_t* bag_size_data,
      const scalar_t* per_sample_weights_data,
      index_t* segment_offsets_data)
      : numel_(numel),
        stride_(stride),
        mode_mean_(mode_mean),
        num_of_segments_(num_of_segments),
        stride_warped_(stride_warped),
        per_sample_weight_defined_(per_sample_weight_defined),
        count_defined_(count_defined),
        per_sample_weights_stride_(per_sample_weights_stride),
        grad_weight_per_segment_data_(grad_weight_per_segment_data),
        indices_data_(indices_data),
        gradoutput_data_(gradoutput_data),
        offset2bag_data_(offset2bag_data),
        count_data_(count_data),
        bag_size_data_(bag_size_data),
        per_sample_weights_data_(per_sample_weights_data),
        segment_offsets_data_(segment_offsets_data) {}

 private:
  int64_t numel_;
  int64_t stride_;
  int mode_mean_;
  int64_t num_of_segments_;
  int64_t stride_warped_;
  bool per_sample_weight_defined_;
  bool count_defined_;
  int64_t per_sample_weights_stride_;
  acc_type_device<scalar_t, kXPU>* grad_weight_per_segment_data_;
  index_t* indices_data_;
  const scalar_t* gradoutput_data_;
  index_t* offset2bag_data_;
  index_t* count_data_;
  index_t* bag_size_data_;
  const scalar_t* per_sample_weights_data_;
  index_t* segment_offsets_data_;
};

template <typename scalar_t, typename index_t>
void compute_grad_weight_bags(
    const Tensor& indices,
    const Tensor& gradOutput,
    const Tensor& offset2bag,
    const Tensor& count,
    int64_t numel,
    int64_t stride,
    int mode_mean,
    const Tensor& bag_size,
    const Tensor& per_sample_weights,
    const Tensor& segment_offsets,
    int64_t num_of_segments,
    const Tensor& grad_weight_per_segment) {
  bool per_sample_weight_defined = per_sample_weights.defined();
  bool count_defined = count.defined();
  int64_t per_sample_weights_stride =
      per_sample_weights.defined() ? per_sample_weights.stride(0) : 0;

  auto grad_weight_per_segment_data =
      grad_weight_per_segment
          .template data_ptr<acc_type_device<scalar_t, kXPU>>();
  auto indices_data = indices.template data_ptr<index_t>();
  auto gradOutput_data = gradOutput.const_data_ptr<scalar_t>();
  auto offset2bag_data = offset2bag.data_ptr<index_t>();
  auto count_data = count_defined
      ? count.data_ptr<index_t>()
      : offset2bag_data; // use the offset2bag_data handler as the dummy
                         // buffer.
  auto bag_size_data = bag_size.data_ptr<index_t>();
  auto per_sample_weights_data = per_sample_weight_defined
      ? per_sample_weights.const_data_ptr<scalar_t>()
      : gradOutput_data; // ise the gradOutput_data handler as the dummy
                         // buffer.
  auto segment_offsets_data = segment_offsets.data_ptr<index_t>();

  int64_t max_sub_group_size = syclMaxSubGroupSize();
  int64_t stride_warped =
      CeilDiv(stride, max_sub_group_size) * max_sub_group_size;

  auto kfn = ComputeGradWeightBagsKernelFunctor<scalar_t, index_t>(
      numel,
      stride,
      mode_mean,
      num_of_segments,
      stride_warped,
      per_sample_weight_defined,
      count_defined,
      per_sample_weights_stride,
      grad_weight_per_segment_data,
      indices_data,
      gradOutput_data,
      offset2bag_data,
      count_data,
      bag_size_data,
      per_sample_weights_data,
      segment_offsets_data);

  int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  int64_t group_size = std::min(stride_warped, work_group_size);
  auto num_groups = CeilDiv(num_of_segments * stride_warped, group_size);
  auto total_items = num_groups * group_size;
  auto global_range = sycl::range<1>((size_t)total_items);
  auto local_range = sycl::range<1>((size_t)group_size);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t, typename index_t>
struct ComputeGradWeightKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto grad_weight_per_segment_ptr = grad_weight_per_segment_data_;
    auto indices_ptr = indices_data_;
    auto grad_output_ptr = grad_output_data_;
    auto count_ptr = count_defined_ ? count_data_ : NULL;
    auto segment_offsets_ptr = segment_offsets_data_;

    const int gid = item.get_global_linear_id();
    const int id = gid / stride_warped_;
    const int startFeature = gid % stride_warped_;
    if (startFeature >= stride_) {
      return;
    }
    if (id >= num_of_segments_) {
      return;
    }
    const int idx_begin = segment_offsets_ptr[id];
    const int idx_end =
        (id == num_of_segments_ - 1) ? numel_ : segment_offsets_ptr[id + 1];

    acc_type_device<scalar_t, kXPU> weight = 0.f;
    for (int idx = idx_begin; idx < idx_end; idx++) {
      const index_t target_row = indices_ptr[idx];

      const acc_type_device<scalar_t, kXPU> scale =
          count_ptr ? 1.f / count_ptr[idx] : 1.f;
      weight += grad_output_ptr[target_row * stride_ + startFeature] * scale;
    }
    grad_weight_per_segment_ptr[id * stride_ + startFeature] = weight;
  }
  ComputeGradWeightKernelFunctor(
      ptrdiff_t numel,
      int64_t stride,
      int64_t num_of_segments,
      int64_t stride_warped,
      bool count_defined,
      acc_type_device<scalar_t, kXPU>* grad_weight_per_segment_data,
      index_t* indices_data,
      const scalar_t* grad_output_data,
      index_t* count_data,
      index_t* segment_offsets_data)
      : numel_(numel),
        stride_(stride),
        num_of_segments_(num_of_segments),
        stride_warped_(stride_warped),
        count_defined_(count_defined),
        grad_weight_per_segment_data_(grad_weight_per_segment_data),
        indices_data_(indices_data),
        grad_output_data_(grad_output_data),
        count_data_(count_data),
        segment_offsets_data_(segment_offsets_data) {}

 private:
  ptrdiff_t numel_;
  int64_t stride_;
  int64_t num_of_segments_;
  int64_t stride_warped_;
  bool count_defined_;
  acc_type_device<scalar_t, kXPU>* grad_weight_per_segment_data_;
  index_t* indices_data_;
  const scalar_t* grad_output_data_;
  index_t* count_data_;
  index_t* segment_offsets_data_;
};

template <typename scalar_t, typename index_t>
void compute_grad_weight(
    const Tensor& indices,
    const Tensor& grad_output,
    const Tensor& count,
    ptrdiff_t numel,
    int64_t stride,
    const Tensor& segment_offsets,
    int64_t num_of_segments,
    const Tensor& grad_weight_per_segment) {
  bool count_defined = count.defined();

  auto grad_weight_per_segment_data =
      grad_weight_per_segment.data_ptr<acc_type_device<scalar_t, kXPU>>();
  auto indices_data = indices.data_ptr<index_t>();
  auto grad_output_data = grad_output.const_data_ptr<scalar_t>();
  auto count_data = count_defined
      ? count.data_ptr<index_t>()
      : indices_data; // use the indices_data handler as the dummy buffer.
  auto segment_offsets_data = segment_offsets.data_ptr<index_t>();

  int64_t max_sub_group_size = syclMaxSubGroupSize();
  int64_t stride_warped =
      CeilDiv(stride, max_sub_group_size) * max_sub_group_size;

  auto kfn = ComputeGradWeightKernelFunctor<scalar_t, index_t>(
      numel,
      stride,
      num_of_segments,
      stride_warped,
      count_defined,
      grad_weight_per_segment_data,
      indices_data,
      grad_output_data,
      count_data,
      segment_offsets_data);

  int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  int64_t group_size = std::min(stride_warped, work_group_size);
  auto num_groups = CeilDiv(num_of_segments * stride_warped, group_size);
  auto total_items = num_groups * group_size;
  auto global_range = sycl::range<1>((size_t)total_items);
  auto local_range = sycl::range<1>((size_t)group_size);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t, typename index_t>
struct SumAndScatterKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto grad_weight_ptr = grad_weight_data_;
    auto input_ptr = input_data_;
    auto segment_offsets_ptr = segment_offsets_data_;
    auto grad_weight_per_segment_ptr = grad_weight_per_segment_data_;
    auto segment_sizes_offsets_ptr = segment_sizes_offsets_data_;

    const int gid = item.get_global_linear_id();
    const int id = gid / stride_warped_;
    const int startFeature = gid % stride_warped_;
    if (startFeature >= stride_) {
      return;
    }
    if (id >= num_of_segments_) {
      return;
    }

    const int idx_begin = segment_sizes_offsets_ptr[id];
    const int idx_end = (id == num_of_segments_ - 1)
        ? num_of_partial_segments_
        : segment_sizes_offsets_ptr[id + 1];
    acc_type_device<scalar_t, kXPU> weight = 0.f;
    for (int idx = idx_begin; idx < idx_end; idx++) {
      weight += grad_weight_per_segment_ptr[idx * stride_ + startFeature];
    }

    int64_t target_row = input_ptr[segment_offsets_ptr[id]];
    if (target_row != padding_idx_) {
      grad_weight_ptr[target_row * stride_ + startFeature] = weight;
    }
  }
  SumAndScatterKernelFunctor(
      int64_t stride,
      int64_t num_of_segments,
      int64_t num_of_partial_segments,
      const int64_t padding_idx,
      int64_t stride_warped,
      scalar_t* grad_weight_data,
      index_t* input_data,
      index_t* segment_offsets_data,
      acc_type_device<scalar_t, kXPU>* grad_weight_per_segment_data,
      index_t* segment_sizes_offsets_data)
      : stride_(stride),
        num_of_segments_(num_of_segments),
        num_of_partial_segments_(num_of_partial_segments),
        padding_idx_(padding_idx),
        stride_warped_(stride_warped),
        grad_weight_data_(grad_weight_data),
        input_data_(input_data),
        segment_offsets_data_(segment_offsets_data),
        grad_weight_per_segment_data_(grad_weight_per_segment_data),
        segment_sizes_offsets_data_(segment_sizes_offsets_data) {}

  void set_stride_warped(int64_t stride_warped) {
    stride_warped_ = stride_warped;
  }

 private:
  int64_t stride_;
  int64_t num_of_segments_;
  int64_t num_of_partial_segments_;
  const int64_t padding_idx_;
  int64_t stride_warped_;
  scalar_t* grad_weight_data_;
  index_t* input_data_;
  index_t* segment_offsets_data_;
  acc_type_device<scalar_t, kXPU>* grad_weight_per_segment_data_;
  index_t* segment_sizes_offsets_data_;
};

template <typename scalar_t, typename index_t>
void sum_and_scatter(
    const Tensor& input,
    const Tensor& grad_weight,
    int64_t stride,
    const Tensor& segment_offsets,
    int64_t num_of_segments,
    const Tensor& grad_weight_per_segment,
    const Tensor& segment_sizes_offsets,
    int64_t num_of_partial_segments,
    const int64_t padding_idx) {
  auto grad_weight_data = grad_weight.data_ptr<scalar_t>();
  auto input_data = input.data_ptr<index_t>();
  auto segment_offsets_data = segment_offsets.data_ptr<index_t>();
  auto grad_weight_per_segment_data =
      grad_weight_per_segment.data_ptr<acc_type_device<scalar_t, kXPU>>();
  auto segment_sizes_offsets_data = segment_sizes_offsets.data_ptr<index_t>();

  auto kfn = SumAndScatterKernelFunctor<scalar_t, index_t>(
      stride,
      num_of_segments,
      num_of_partial_segments,
      padding_idx,
      /* stride_warped */ 0,
      grad_weight_data,
      input_data,
      segment_offsets_data,
      grad_weight_per_segment_data,
      segment_sizes_offsets_data);

  int64_t work_group_size = syclMaxWorkGroupSize(kfn);
  int64_t stride_warped = CeilDiv(stride, work_group_size) * work_group_size;
  kfn.set_stride_warped(stride_warped);

  int64_t group_size = std::min(stride_warped, work_group_size);
  auto num_groups = CeilDiv(num_of_segments * stride_warped, group_size);
  auto total_items = num_groups * group_size;
  auto global_range = sycl::range<1>((size_t)total_items);
  auto local_range = sycl::range<1>((size_t)group_size);

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

struct EmbeddingBackwardDeterministicKernelCopyIfFunctor {
  template <typename T>
  auto operator()(T x) const {
    return x != -1;
  }
};

struct EmbeddingBackwardDeterministicKernelTransformFirstTrueFunctor {
  template <typename U, typename V>
  auto operator()(U d, V idx) const {
    return d ? idx : -1;
  }
};

struct EmbeddingBackwardDeterministicKernelAdjacentDifferenceFunctor {
  template <typename T>
  bool operator()(T lhs, T rhs) const {
    if (lhs != rhs) {
      return true;
    }
    return false;
  }
};

template <typename scalar_t, typename index_t>
Tensor embedding_backward_deterministic_kernel(
    const Tensor& grad,
    const Tensor& orig_indices,
    const Tensor& sorted_indices,
    const Tensor& count,
    int64_t num_weights,
    int64_t padding_idx = -1,
    bool mode_mean = false,
    const Tensor& offset2bag = Tensor(),
    const Tensor& bag_size = Tensor(),
    const Tensor& per_sample_weights = Tensor()) {
  const int64_t numel = sorted_indices.numel();
  auto grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());
  const int64_t stride = grad_weight.stride(0);

  auto segment_offsets = at::empty({numel}, orig_indices.options());
  index_t num_of_segments;
  {
    // sorted:          2 5 5 5 7 7 8 9 9
    // dummy:           1 1 0 0 1 0 1 1 0
    // segment_offsets: 0 1 - - 4 - 6 7 -
    auto sorted_indices_begin = sorted_indices.data_ptr<index_t>();
    auto dummy = at::empty_like(sorted_indices);
    auto dummy_begin = dummy.data_ptr<index_t>();
    auto idx_tensor = at::empty_like(sorted_indices);
    auto idx_begin = idx_tensor.data_ptr<index_t>();
    EmbeddingBackwardDeterministicKernelAdjacentDifferenceFunctor
        adjacent_difference_functor;
    pstl::adjacent_difference<index_t>(
        sorted_indices_begin,
        sorted_indices_begin + numel,
        dummy_begin,
        adjacent_difference_functor);

    // For algorithm adjacent difference, for output, its first element is
    // always equal to source first element. We need to set it as 1 manually.
    Tensor count_tensor =
        at::empty({numel}, at::TensorOptions().device(kXPU).dtype(kLong));
    auto count_begin = count_tensor.data_ptr<int64_t>();
    pstl::itoa(count_begin, count_begin + numel, (int64_t)0);
    auto segment_offsets_begin = segment_offsets.data_ptr<index_t>();
    EmbeddingBackwardDeterministicKernelTransformFirstTrueFunctor
        transform_first_true_functor;
    pstl::transform_first_true<index_t>(
        dummy_begin,
        dummy_begin + numel,
        count_begin,
        idx_begin,
        transform_first_true_functor);
    EmbeddingBackwardDeterministicKernelCopyIfFunctor copy_if_functor;
    auto ends = pstl::copy_if<index_t>(
        idx_begin, idx_begin + numel, segment_offsets_begin, copy_if_functor);
    num_of_segments = std::distance(segment_offsets_begin, ends);
  }

  auto partials_per_segment =
      at::empty({num_of_segments}, orig_indices.options());

  krn_partials_per_segment<index_t>(
      partials_per_segment.template data_ptr<index_t>(),
      segment_offsets.data_ptr<index_t>(),
      num_of_segments,
      numel);

  // In order to compute `partial_segment_offset`, which is the start index
  // of each partial-segment in `sorted_indices`, we need to compute the
  // start position of each _segment_ in `partial_segment_offset`.
  // Unit: index in `partial_segment_offset`
  auto partials_per_segment_offset =
      at::empty({num_of_segments}, orig_indices.options());
  pstl::exclusive_scan(
      partials_per_segment.template data_ptr<index_t>(),
      partials_per_segment.template data_ptr<index_t>() + num_of_segments,
      partials_per_segment_offset.template data_ptr<index_t>(),
      (index_t)0);

  // The total number of partial-segments is the sum of
  // `partials_per_segment_offset`
  auto num_of_partial_segments =
      partials_per_segment[num_of_segments - 1].template item<index_t>() +
      partials_per_segment_offset[num_of_segments - 1].template item<index_t>();

  auto partial_segment_offset =
      at::empty({num_of_partial_segments}, orig_indices.options());
  krn_partial_segment_offset<index_t>(
      partial_segment_offset.template data_ptr<index_t>(),
      partials_per_segment.template data_ptr<index_t>(),
      partials_per_segment_offset.template data_ptr<index_t>(),
      segment_offsets.data_ptr<index_t>(),
      num_of_segments);

  TensorOptions op;
  if (grad.dtype() == at::kBFloat16 || grad.dtype() == at::kHalf) {
    op = grad.options().dtype(at::kFloat);
  } else {
    op = grad.options();
  }
  auto grad_weight_per_segment =
      at::empty({num_of_partial_segments, stride}, op);
  // Compute the sum of each partial-segment and handle bags
  if (offset2bag.defined()) {
    compute_grad_weight_bags<scalar_t, index_t>(
        orig_indices,
        grad,
        offset2bag,
        count,
        numel,
        stride,
        mode_mean,
        bag_size,
        per_sample_weights,
        partial_segment_offset,
        num_of_partial_segments,
        grad_weight_per_segment);
  } else {
    compute_grad_weight<scalar_t, index_t>(
        orig_indices,
        grad,
        count,
        numel,
        stride,
        partial_segment_offset,
        num_of_partial_segments,
        grad_weight_per_segment);
  }

  sum_and_scatter<scalar_t, index_t>(
      sorted_indices,
      grad_weight,
      stride,
      segment_offsets,
      num_of_segments,
      grad_weight_per_segment,
      partials_per_segment_offset,
      num_of_partial_segments,
      padding_idx);

  return grad_weight;
}

} // namespace at::native::xpu
