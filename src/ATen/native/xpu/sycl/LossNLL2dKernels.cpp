/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <comm/Macros.h>
// clang-format off
DISABLE_RETURN_TYPE_WARNING_BEGIN
// clang-format on
#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/LossNLL2dKernels.h>

namespace at::native::xpu {
inline Tensor optional_contiguous(const Tensor& source) {
  return source.defined() ? source.contiguous() : source;
}

template <typename scalar_t>
inline const scalar_t* optional_data(const Tensor& source) {
  return source.defined() ? source.const_data_ptr<scalar_t>() : nullptr;
}

template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss2d_forward_noreduce_kernel(
    int64_t n_threads,
    PackedTensorAccessor64<scalar_t, 4> input,
    PackedTensorAccessor64<int64_t, 3> target,
    PackedTensorAccessor64<scalar_t, 3> output,
    const scalar_t* weight,
    int64_t ignore_index) {
  int64_t batch_size = input.size(0);
  int64_t n_classes = input.size(1);
  int64_t H = input.size(2);
  int64_t W = input.size(3);

  auto item = syclext::this_work_item::get_nd_item<1>();
  int64_t linear_id =
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  PackedTensorAccessor64<scalar_t, 3> output_result = output;
  for (int32_t index = linear_id; linear_id < (n_threads);
       linear_id += item.get_group_range(0) * item.get_local_range(0),
               index = linear_id) {
    const int64_t b = index % batch_size;
    const int64_t h = (index / batch_size) % H;
    const int64_t w = (index / (batch_size * H)) % W;

    int64_t cur_target = target[b][h][w];
    if (cur_target == ignore_index) {
      output_result[b][h][w] = static_cast<scalar_t>(0);
      continue;
    }
    SYCL_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
    scalar_t value = input[b][cur_target][h][w];
    scalar_t cur_weight =
        weight != nullptr ? weight[cur_target] : static_cast<scalar_t>(1);
    output_result[b][h][w] = -value * cur_weight;
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t, int SIMD>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::sub_group_size<SIMD>)) void nll_loss2d_forward_kernel_(
    scalar_t* output,
    scalar_t* total_weight,
    const scalar_t* input,
    const int64_t* target,
    const scalar_t* weight,
    int n_classes,
    int map_nelem,
    int blocks_per_sample,
    int64_t ignore_index,
    int64_t work_group_size) {
  scalar_t cur_weight;
  accscalar_t input_sum = 0;
  accscalar_t acc_weight = 0;

  auto item = syclext::this_work_item::get_nd_item<1>();

  index_t sample = item.get_group(0) / blocks_per_sample;
  index_t toffset = sample * map_nelem;
  index_t ioffset = sample * map_nelem * n_classes;
  int step = item.get_local_range(0) * blocks_per_sample;
  auto start =
      (item.get_group(0) % blocks_per_sample) * item.get_local_range(0) +
      item.get_local_id(0);
  for (int i = start; i < map_nelem; i += step) {
    index_t t = target[toffset + i];
    if (t != ignore_index) {
      SYCL_KERNEL_ASSERT(t >= 0 && t < n_classes);
      cur_weight = weight != nullptr ? weight[t] : static_cast<scalar_t>(1);
      const auto input_index = ioffset + i + map_nelem * t;
      SYCL_KERNEL_ASSERT(input_index >= 0);
      input_sum -= input[input_index] * cur_weight;
      acc_weight += cur_weight;
    }
  }
  char* lsm = (char*)syclexp::get_work_group_scratch_memory();
  auto acc_weight_smem = reinterpret_cast<accscalar_t*>(lsm);
  auto input_sum_smem = reinterpret_cast<accscalar_t*>(
      lsm + sizeof(accscalar_t) * work_group_size);

  auto acc_weight_reduce = GroupReduceSumWithoutBroadcast<accscalar_t, SIMD>(
      item, acc_weight, acc_weight_smem);
  auto input_sum_reduce = GroupReduceSumWithoutBroadcast<accscalar_t, SIMD>(
      item, input_sum, input_sum_smem);

  if (item.get_local_id(0) == 0) {
    atomicAdd(
        sycl_global_ptr<scalar_t>(total_weight),
        static_cast<scalar_t>(acc_weight_reduce));
    atomicAdd(
        sycl_global_ptr<scalar_t>(output),
        static_cast<scalar_t>(input_sum_reduce));
  }
}

template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss2d_forward_average_kernel(
    scalar_t* output,
    const scalar_t* total_weight) {
  *output /= *total_weight;
}

void nll_loss2d_forward_kernel(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  if (reduction != at::Reduction::None) {
    at::globalContext().alertNotDeterministic("nll_loss2d_forward_xpu");
  }

  total_weight.resize_({});
  total_weight.zero_();

  if (reduction == at::Reduction::None) {
    int64_t batch_size = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t count = batch_size * H * W;

    at::native::resize_output(output, {batch_size, H, W});
    if (count == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with
      // 0 blocks.
      return;
    }
    auto weight_ = optional_contiguous(weight);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss2d_forward_no_reduce_kernel",
        [&] {
          constexpr auto kfn = nll_loss2d_forward_noreduce_kernel<scalar_t>;
          int64_t local_range = syclMaxWorkGroupSize<kfn>();
          auto global_range = (count + local_range - 1) / local_range;
          sycl_kernel_submit<kfn>(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              0,
              count,
              input.packed_accessor64<scalar_t, 4>(),
              target.packed_accessor64<int64_t, 3>(),
              output.packed_accessor64<scalar_t, 3>(),
              optional_data<scalar_t>(weight_),
              ignore_index);
        });
    return;
  }

  // produce scalar outputs for the reduction case
  at::native::resize_output(output, {});

  if (target.numel() == 0) {
    if (reduction == Reduction::Mean) {
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      output.zero_();
    }
    return;
  }

  auto input_ = input.contiguous();
  auto weight_ = optional_contiguous(weight);
  auto target_ = target.contiguous();

  output.zero_();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "nll_loss2d_forward_kernel",
      [&input_,
       &weight_,
       &target_,
       &output,
       &total_weight,
       &input,
       &target,
       &reduction,
       &ignore_index] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        AT_DISPATCH_INDEX_TYPES(
            at::native::canUse32BitIndexMath(input_, INT_MAX)
                ? ScalarType::Int
                : ScalarType::Long,
            "nll_loss2d_forward_launcher",
            [&] {
              auto batch_size = target.size(0);
              int64_t map_nelem = target.numel() / batch_size;
              const int simd = 32;
              int64_t work_group_size = get_group_reduce_group_size(simd);
              int blocks_per_sample =
                  (map_nelem + work_group_size - 1) / work_group_size / 128;
              blocks_per_sample =
                  (blocks_per_sample == 0) ? 1 : blocks_per_sample;
              int total_blocks = blocks_per_sample * batch_size;
              constexpr auto kfn = nll_loss2d_forward_kernel_<
                  scalar_t,
                  accscalar_t,
                  index_t,
                  simd>;
              sycl_kernel_submit<kfn>(
                  total_blocks * work_group_size,
                  work_group_size,
                  getCurrentSYCLQueue(),
                  2 * sizeof(accscalar_t) * work_group_size,
                  output.mutable_data_ptr<scalar_t>(),
                  total_weight.mutable_data_ptr<scalar_t>(),
                  input_.const_data_ptr<scalar_t>(),
                  target_.const_data_ptr<int64_t>(),
                  optional_data<scalar_t>(weight_),
                  input_.size(1),
                  input_.size(2) * input_.size(3),
                  blocks_per_sample,
                  ignore_index,
                  work_group_size);
              // Divide by total_weight
              if (reduction == at::Reduction::Mean) {
                constexpr auto kfn =
                    nll_loss2d_forward_average_kernel<scalar_t>;
                sycl_kernel_submit<kfn>(
                    1,
                    1,
                    getCurrentSYCLQueue(),
                    0,
                    output.mutable_data_ptr<scalar_t>(),
                    total_weight.const_data_ptr<scalar_t>());
              }
            });
      });
}

template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss2d_backward_noreduce_kernel(
    int64_t n_threads,
    PackedTensorAccessor64<int64_t, 3> target,
    PackedTensorAccessor64<scalar_t, 3> grad_output,
    PackedTensorAccessor64<scalar_t, 4> grad_input,
    const scalar_t* weight,
    int64_t ignore_index) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int64_t batch_size = target.size(0);
  int64_t H = target.size(1);
  int64_t W = target.size(2);

  auto grad_input_result = grad_input;
  int64_t linear_id =
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  for (int32_t index = linear_id; linear_id < (n_threads);
       linear_id += item.get_group_range(0) * item.get_local_range(0),
               index = linear_id) {
    const int64_t b = index % batch_size;
    const int64_t h = (index / batch_size) % H;
    const int64_t w = (index / (batch_size * H)) % W;

    int64_t cur_target = target[b][h][w];
    if (cur_target == ignore_index) {
      continue;
    }
    scalar_t value =
        -(weight != nullptr ? weight[cur_target] : static_cast<scalar_t>(1));
    grad_input_result[b][cur_target][h][w] = value * grad_output[b][h][w];
  }
}

template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss2d_backward_kernel_(
    scalar_t* grad_input,
    const scalar_t* grad_output,
    const int64_t* target,
    const scalar_t* weights,
    const scalar_t* total_weight,
    bool size_average,
    int n_classes,
    int map_nelem,
    int blocks_per_sample,
    int64_t ignore_index) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  const auto grad =
      -(size_average ? *grad_output / *total_weight : *grad_output);

  const int sample = item.get_group(0) / blocks_per_sample;
  const int step = item.get_local_range(0) * blocks_per_sample;

  const int toffset = sample * map_nelem;
  const auto* const target_thread = target + toffset;

  const int ioffset = sample * map_nelem * n_classes;
  auto* const grad_input_thread = grad_input + ioffset;

  for (int i =
           (item.get_group(0) % blocks_per_sample) * item.get_local_range(0) +
           item.get_local_id(0);
       i < map_nelem;
       i += step) {
    const int64_t t = target_thread[i];
    if (t != ignore_index) {
      SYCL_KERNEL_ASSERT(t >= 0 && t < n_classes);
      const auto grad_input_index = i + map_nelem * t;
      SYCL_KERNEL_ASSERT(grad_input_index >= 0);
      grad_input_thread[i + map_nelem * t] =
          weights != nullptr ? weights[t] * grad : grad;
    }
  }
}

void nll_loss2d_backward_kernel(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {
  grad_input.resize_as_(input);
  grad_input.zero_();
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(
      total_weight.numel() == 1,
      "expected total_weight to be a single element tensor, got: ",
      total_weight.sizes(),
      " (",
      total_weight.numel(),
      " elements)");

  if (reduction == at::Reduction::None) {
    TORCH_CHECK(
        grad_output.dim() == 3,
        "grad_output must have same dimension as target (3) but got dimension: ",
        grad_output.sizes());
    TORCH_CHECK(
        grad_output.size(0) == target.size(0) &&
            grad_output.size(1) == target.size(1) &&
            grad_output.size(2) == target.size(2),
        "grad_output sizes don't match target sizes: target ",
        target.sizes(),
        ", grad_output ",
        grad_output.sizes())

    int64_t batch_size = input.size(0);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t count = batch_size * H * W;

    if (count == 0) {
      // This guards from unnecessary operations and launching kernel with
      // 0 blocks.
      return;
    }

    auto weight_ = optional_contiguous(weight);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss2d_backward_no_reduce_kernel",
        [&] {
          constexpr auto kfn = nll_loss2d_backward_noreduce_kernel<scalar_t>;
          int64_t local_range = syclMaxWorkGroupSize<kfn>();
          auto global_range = (count + local_range - 1) / local_range;
          sycl_kernel_submit<kfn>(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              0,
              count,
              target.packed_accessor64<int64_t, 3>(),
              grad_output.packed_accessor64<scalar_t, 3>(),
              grad_input.packed_accessor64<scalar_t, 4>(),
              optional_data<scalar_t>(weight_),
              ignore_index);
        });
    return;
  }

  int64_t batch_size = target.size(0);
  auto target_numel = target.numel();
  if (batch_size != 0 && target_numel != 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss2d_backward_kernel",
        [&] {
          // This guards from unnecessary operations and launching kernel with 1
          // blocks.
          auto target_ = target.contiguous();
          auto weight_ = optional_contiguous(weight);
          int64_t map_nelem = target_numel / batch_size;
          constexpr auto kfn = nll_loss2d_backward_kernel_<scalar_t>;
          int64_t max_work_group_size = syclMaxWorkGroupSize<kfn>();
          int blocks_per_sample =
              (map_nelem + max_work_group_size - 1) / max_work_group_size / 128;
          blocks_per_sample = (blocks_per_sample == 0) ? 1 : blocks_per_sample;
          int total_blocks = blocks_per_sample * batch_size;
          sycl_kernel_submit<kfn>(
              total_blocks * max_work_group_size,
              max_work_group_size,
              getCurrentSYCLQueue(),
              0,
              grad_input.mutable_data_ptr<scalar_t>(),
              grad_output.const_data_ptr<scalar_t>(),
              target_.const_data_ptr<int64_t>(),
              optional_data<scalar_t>(weight_),
              total_weight.const_data_ptr<scalar_t>(),
              reduction == at::Reduction::Mean,
              input.size(1),
              map_nelem,
              blocks_per_sample,
              ignore_index);
        });
  }
}

} // namespace at::native::xpu

// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
