/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/LossNLLKernel.h>

#include <comm/TensorInfo.h>

namespace at::native::xpu {

#define CHECK_INDEX_IN_CLASS(INDEX, N_CLASSES)           \
  if constexpr (std::is_unsigned_v<decltype(INDEX)>) {   \
    SYCL_KERNEL_ASSERT(INDEX < N_CLASSES);               \
  } else {                                               \
    SYCL_KERNEL_ASSERT(INDEX >= 0 && INDEX < N_CLASSES); \
  }

int nll_loss_threads(int64_t nframe) {
  return std::clamp(
      1 << static_cast<int64_t>(std::round(std::log2(nframe / 16))), 32, 1024);
}

using namespace at::xpu;

template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss_forward_no_reduce_kernel(
    int64_t batch_size,
    PackedTensorAccessor64<scalar_t, 2> input,
    const index_t* target,
    scalar_t* output,
    const scalar_t* weights,
    int64_t n_classes,
    int64_t ignore_index) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, batch_size) {
    index_t cur_target = target[index];
    if (cur_target == ignore_index) {
      output[index] = static_cast<scalar_t>(0);
      continue;
    }
    CHECK_INDEX_IN_CLASS(cur_target, n_classes);
    auto cur_weight =
        weights != nullptr ? weights[cur_target] : static_cast<scalar_t>(1);
    output[index] = -cur_weight * input[index][cur_target];
  }
}

template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss_forward_reduce1d_kernel(
    scalar_t* output,
    scalar_t* total_weight,
    const scalar_t* input,
    const index_t* target,
    const scalar_t* weights,
    bool size_average,
    int64_t n_classes,
    int64_t ignore_index) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  SYCL_KERNEL_ASSERT(item.get_local_id(0) == 0 && item.get_group(0) == 0);

  const index_t t = *target;
  if (t != ignore_index) {
    CHECK_INDEX_IN_CLASS(t, n_classes);
    const auto cur_weight = weights != nullptr ? weights[t] : scalar_t{1};
    *total_weight = cur_weight;

    if (size_average) {
      // If we try to normalize a zero then we return a NaN
      if (cur_weight == 0) {
        *output = std::numeric_limits<scalar_t>::quiet_NaN();
      } else {
        *output = -input[t];
      }
    } else {
      *output = -cur_weight * input[t];
    }
  } else {
    *output = scalar_t{0};
    *total_weight = scalar_t{0};
  }
}

template <typename scalar_t, typename index_t, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss_forward_reduce2d_kernel(
    scalar_t* output,
    scalar_t* total_weight,
    const scalar_t* input,
    const index_t* target,
    const scalar_t* weights,
    bool size_average,
    int64_t nframe,
    int64_t ndim,
    int64_t n_classes,
    int64_t ignore_index,
    int64_t smem_size) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  char* scratch = (char*)syclexp::get_work_group_scratch_memory();
  accscalar_t* sh_inputs = (accscalar_t*)scratch;
  accscalar_t* acc_weight =
      (accscalar_t*)(scratch + sizeof(accscalar_t) * smem_size);

  auto local_id = item.get_local_id(0);
  auto local_range = item.get_local_range(0);

  sh_inputs[local_id] = static_cast<accscalar_t>(0);
  acc_weight[local_id] = static_cast<accscalar_t>(0);

  for (int i = local_id; i < nframe; i += local_range) {
    index_t t = target[i];
    if (t != ignore_index) {
      CHECK_INDEX_IN_CLASS(t, n_classes);
      scalar_t cur_weight =
          weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
      sh_inputs[local_id] -= input[i * ndim + t] * cur_weight;
      acc_weight[local_id] += cur_weight;
    }
  }

  sycl::group_barrier(item.get_group());

  for (int stride = local_range / 2; stride > 0; stride >>= 1) {
    if (local_id < stride) {
      sh_inputs[local_id] += sh_inputs[local_id + stride];
      acc_weight[local_id] += acc_weight[local_id + stride];
    }
    sycl::group_barrier(item.get_group());
  }

  if (local_id == 0) {
    *total_weight = static_cast<scalar_t>(acc_weight[0]);
    if (size_average) {
      *output = static_cast<scalar_t>(sh_inputs[0] / acc_weight[0]);
    } else {
      *output = static_cast<scalar_t>(sh_inputs[0]);
    }
  }
}

template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss_backward_no_reduce_kernel(
    int batch_size,
    const index_t* target,
    PackedTensorAccessor64<const scalar_t, 1> grad_output,
    PackedTensorAccessor64<scalar_t, 2> grad_input,
    const scalar_t* weights,
    int64_t n_classes,
    int64_t ignore_index) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  XPU_KERNEL_LOOP(item, index, batch_size) {
    index_t cur_target = target[index];
    if (cur_target == ignore_index) {
      continue;
    }
    CHECK_INDEX_IN_CLASS(cur_target, n_classes);
    scalar_t weight =
        weights != nullptr ? weights[cur_target] : static_cast<scalar_t>(1);
    grad_input[index][cur_target] = -weight * grad_output[index];
  }
}

template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss_backward_reduce1d_kernel(
    scalar_t* grad_input,
    const scalar_t* grad_output,
    const scalar_t* weights,
    const index_t* target,
    const scalar_t* total_weight,
    bool size_average,
    int64_t n_classes,
    int64_t ignore_index) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  SYCL_KERNEL_ASSERT(item.get_local_id(0) == 0 && item.get_group(0) == 0);
  const index_t t = *target;
  if (t != ignore_index) {
    CHECK_INDEX_IN_CLASS(t, n_classes);
    const auto grad =
        -(size_average ? *grad_output / *total_weight : *grad_output);
    grad_input[t] = weights != nullptr ? weights[t] * grad : grad;
  }
}

template <typename T>
struct bwd_index_type {
  using type = T;
};
template <>
struct bwd_index_type<uint8_t> {
  using type = int;
};
template <>
struct bwd_index_type<int64_t> {
  using type = uint64_t;
};

template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void nll_loss_backward_reduce2d_kernel(
    scalar_t* grad_input,
    const scalar_t* grad_output,
    const index_t* target,
    const scalar_t* weights,
    const scalar_t* total_weight,
    bool size_average,
    int nframe,
    int ndim,
    int64_t n_classes,
    int64_t ignore_index) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  auto local_id = item.get_local_id(0);
  auto local_range = item.get_local_range(0);
  using bwd_index_t = typename bwd_index_type<index_t>::type;
  const auto grad =
      -(size_average ? *grad_output / *total_weight : *grad_output);

  for (int i = local_id; i < nframe; i += local_range) {
    const index_t t = target[i];
    if (t != ignore_index) {
      CHECK_INDEX_IN_CLASS(t, n_classes);
      const bwd_index_t index = static_cast<bwd_index_t>(i) * ndim + t;
      if constexpr (!std::is_unsigned_v<decltype(index)>) {
        SYCL_KERNEL_ASSERT(index >= 0);
      }
      grad_input[index] = weights != nullptr ? weights[t] * grad : grad;
    }
  }
}

#define AT_DISPATCH_NLL_LOSS_INDEX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                     \
      TYPE,                                               \
      NAME,                                               \
      AT_PRIVATE_CASE_TYPE_USING_HINT(                    \
          at::ScalarType::Byte, index_t, __VA_ARGS__)     \
          AT_PRIVATE_CASE_TYPE_USING_HINT(                \
              at::ScalarType::Long, index_t, __VA_ARGS__))

void nll_loss_forward_kernel(
    const Tensor& output,
    const Tensor& total_weight,
    const Tensor& input_,
    const Tensor& target_,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto input = *input_.expect_contiguous();
  auto target = *target_.expect_contiguous();

  int64_t n_classes = input.size(-1);
  int64_t n_dims = input.dim();
  int64_t batch_size = n_dims == 1 ? 1 : input.size(0);

  auto weight_ = weight.defined() ? weight.contiguous() : weight;

  if (weight_.defined()) {
    TORCH_CHECK(
        input.scalar_type() == weight_.scalar_type(),
        "expected scalar type ",
        input.scalar_type(),
        " but found ",
        weight_.scalar_type());
  }

  if (reduction == at::Reduction::None && n_dims == 2) {
    at::native::resize_output(output, {batch_size});
    total_weight.zero_();
    if (batch_size == 0) {
      // This guards from unnecessary operations and launching SYCL kernel with
      // 0 blocks.
      return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_forward_no_reduce_xpu_kernel",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_forward_no_reduce_xpu_kernel_index",
              [&] {
                sycl_kernel_submit<
                    nll_loss_forward_no_reduce_kernel<scalar_t, index_t>>(
                    xpuKernelLoopGroupRange(batch_size, 1024) * 1024,
                    1024,
                    getCurrentSYCLQueue(),
                    0,
                    batch_size,
                    input.packed_accessor64<scalar_t, 2>(),
                    target.const_data_ptr<index_t>(),
                    output.mutable_data_ptr<scalar_t>(),
                    weight_.defined() ? weight_.const_data_ptr<scalar_t>()
                                      : nullptr,
                    n_classes,
                    ignore_index);
              });
        });
    return;
  }

  // produce scalar outputs for the reduction case
  at::native::resize_output(output, {});
  total_weight.resize_({});

  if (target.numel() == 0) {
    if (reduction == at::Reduction::Mean) {
      output.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      output.zero_();
    }
    total_weight.zero_();
    return;
  }

  if (n_dims == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_forward_reduce_xpu_kernel_1d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_forward_reduce_xpu_kernel_1d_index",
              [&] {
                sycl_kernel_submit<
                    nll_loss_forward_reduce1d_kernel<scalar_t, index_t>>(
                    1,
                    1,
                    getCurrentSYCLQueue(),
                    0,
                    output.mutable_data_ptr<scalar_t>(),
                    total_weight.mutable_data_ptr<scalar_t>(),
                    input.const_data_ptr<scalar_t>(),
                    target.const_data_ptr<index_t>(),
                    weight_.defined() ? weight_.const_data_ptr<scalar_t>()
                                      : nullptr,
                    reduction == at::Reduction::Mean,
                    n_classes,
                    ignore_index);
              });
        });

  } else if (n_dims == 2) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_forward_reduce_xpu_kernel_2d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_forward_reduce_xpu_kernel_2d_index",
              [&] {
                using accscalar_t = at::acc_type<scalar_t, true>;
                int nthreads = nll_loss_threads(input.size(0));
                size_t slm_sz = 2 * sizeof(accscalar_t) * nthreads;
                sycl_kernel_submit<nll_loss_forward_reduce2d_kernel<
                    scalar_t,
                    index_t,
                    accscalar_t>>(
                    sycl::range<1>(nthreads),
                    sycl::range<1>(nthreads),
                    getCurrentSYCLQueue(),
                    slm_sz,
                    output.mutable_data_ptr<scalar_t>(),
                    total_weight.mutable_data_ptr<scalar_t>(),
                    input.const_data_ptr<scalar_t>(),
                    target.const_data_ptr<index_t>(),
                    weight_.defined() ? weight_.const_data_ptr<scalar_t>()
                                      : nullptr,
                    reduction == at::Reduction::Mean,
                    input.size(0),
                    input.size(1),
                    n_classes,
                    ignore_index,
                    (int64_t)nthreads);
              });
        });
  }
}

void nll_loss_backward_kernel(
    const Tensor& grad_input_,
    const Tensor& grad_output_,
    const Tensor& input_,
    const Tensor& target_,
    const Tensor& total_weight,
    const Tensor& weight,
    int64_t reduction,
    int64_t ignore_index) {
  auto target = *target_.expect_contiguous();
  auto input = *input_.expect_contiguous();
  auto grad_input = *grad_input_.expect_contiguous();
  auto grad_output = *grad_output_.expect_contiguous();

  int64_t n_dims = input.dim();
  int64_t n_classes = input.size(-1);
  int64_t batch_size = n_dims == 1 ? 1 : input.size(0);

  auto weight_ = weight.defined() ? weight.contiguous() : weight;

  if (reduction == at::Reduction::None && n_dims == 2) {
    if (batch_size == 0) {
      return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_backward_no_reduce_xpu_kernel",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_backward_no_reduce_xpu_kernel_index",
              [&] {
                sycl_kernel_submit<
                    nll_loss_backward_no_reduce_kernel<scalar_t, index_t>>(
                    xpuKernelLoopGroupRange(batch_size, 1024) * 1024,
                    1024,
                    getCurrentSYCLQueue(),
                    0,
                    batch_size,
                    target.const_data_ptr<index_t>(),
                    grad_output.packed_accessor64<const scalar_t, 1>(),
                    grad_input.packed_accessor64<scalar_t, 2>(),
                    weight.defined() ? weight_.const_data_ptr<scalar_t>()
                                     : nullptr,
                    n_classes,
                    ignore_index);
              });
        });
    return;
  }

  if (n_dims == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_backward_reduce_xpu_kernel_1d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_backward_reduce_xpu_kernel_1d_index",
              [&] {
                sycl_kernel_submit<
                    nll_loss_backward_reduce1d_kernel<scalar_t, index_t>>(
                    sycl::range<1>(1),
                    sycl::range<1>(1),
                    getCurrentSYCLQueue(),
                    0,
                    grad_input.mutable_data_ptr<scalar_t>(),
                    grad_output.const_data_ptr<scalar_t>(),
                    weight.defined() ? weight_.const_data_ptr<scalar_t>()
                                     : nullptr,
                    target.const_data_ptr<index_t>(),
                    total_weight.const_data_ptr<scalar_t>(),
                    reduction == at::Reduction::Mean,
                    n_classes,
                    ignore_index);
              });
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_backward_reduce_xpu_kernel_2d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_backward_reduce_xpu_kernel_2d_index",
              [&] {
                sycl_kernel_submit<
                    nll_loss_backward_reduce2d_kernel<scalar_t, index_t>>(
                    nll_loss_threads(input.size(0)),
                    nll_loss_threads(input.size(0)),
                    getCurrentSYCLQueue(),
                    0,
                    grad_input.mutable_data_ptr<scalar_t>(),
                    grad_output.const_data_ptr<scalar_t>(),
                    target.const_data_ptr<index_t>(),
                    weight.defined() ? weight_.const_data_ptr<scalar_t>()
                                     : nullptr,
                    total_weight.const_data_ptr<scalar_t>(),
                    reduction == at::Reduction::Mean,
                    input.size(0),
                    input.size(1),
                    n_classes,
                    ignore_index);
              });
        });
  }
}

#undef AT_DISPATCH_NLL_LOSS_INDEX_TYPES
} // namespace at::native::xpu
