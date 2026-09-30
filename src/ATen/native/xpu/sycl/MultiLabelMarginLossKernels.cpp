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
#include <ATen/core/Tensor.h>
#include <ATen/native/LossMulti.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/MultiLabelMarginLossKernels.h>

namespace at::native::xpu {

const int MULTILABELMARGIN_SUB_GROUP_SIZE = 32;
const int MULTILABELMARGIN_THREADS =
    MULTILABELMARGIN_SUB_GROUP_SIZE * MULTILABELMARGIN_SUB_GROUP_SIZE;

using namespace at::xpu;

template <typename scalar_t, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::sub_group_size<MULTILABELMARGIN_SUB_GROUP_SIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void multilabel_margin_loss_forward_kernel_impl(
    scalar_t* output,
    const scalar_t* input,
    const int64_t* target,
    scalar_t* is_target,
    int nframe,
    int dim,
    bool size_average) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  accscalar_t* smem =
      reinterpret_cast<accscalar_t*>(syclexp::get_work_group_scratch_memory());

  int k = item.get_group(0);
  const scalar_t* input_k = input + k * dim;
  const int64_t* target_k = target + k * dim;
  scalar_t* output_k = output + k;
  scalar_t* is_target_k = is_target + k * dim;
  for (int d = item.get_local_linear_id(); d < dim;
       d += item.get_local_range(0)) {
    is_target_k[d] = static_cast<scalar_t>(0);
  }
  sycl::group_barrier(item.get_group());

  if (item.get_local_linear_id() == 0) {
    for (int dt = 0; dt < dim; dt++) {
      int target_idx = target_k[dt];
      if (target_idx < 0) {
        break;
      }
      is_target_k[target_idx] = static_cast<scalar_t>(1);
    }
  }
  sycl::group_barrier(item.get_group());

  accscalar_t sum = 0;
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = target_k[dt];
    if (target_idx < 0) {
      break;
    }

    // current value for target
    scalar_t input_target_k = input_k[target_idx];

    // compare to all inputs (multithreaded):
    for (int d = item.get_local_linear_id(); d < dim;
         d += item.get_local_range(0)) {
      // contribute to loss only if not a target
      if (!static_cast<int>(is_target_k[d])) {
        scalar_t z = 1 - input_target_k + input_k[d];
        if (z > 0) {
          sum += z;
        }
      }
    }
  }

  accscalar_t total_sum = GroupReduceSumWithoutBroadcast<
      accscalar_t,
      MULTILABELMARGIN_SUB_GROUP_SIZE>(item, sum, smem);

  if (item.get_local_linear_id() == 0) {
    if (size_average) {
      *output_k = static_cast<scalar_t>((total_sum / dim) / nframe);
    } else {
      *output_k = static_cast<scalar_t>(total_sum / dim);
    }
  }
}

template <typename scalar_t, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY(
    (syclexp::sub_group_size<MULTILABELMARGIN_SUB_GROUP_SIZE>))
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void multilabel_margin_loss_backward_kernel_impl(
    scalar_t* grad_input,
    const scalar_t* grad_output,
    const scalar_t* input,
    const int64_t* target,
    const scalar_t* is_target,
    int nframe,
    int dim,
    bool size_average,
    bool reduce) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  accscalar_t* smem =
      reinterpret_cast<accscalar_t*>(syclexp::get_work_group_scratch_memory());

  int k = item.get_group(0);
  const scalar_t* input_k = input + k * dim;
  scalar_t* grad_input_k = grad_input + k * dim;
  const int64_t* target_k = target + k * dim;
  const scalar_t* is_target_k = is_target + k * dim;

  const scalar_t* grad_output_k = grad_output;
  if (!reduce) {
    grad_output_k += k;
  }

  // gain:
  scalar_t g = static_cast<scalar_t>(
      size_average && reduce
          ? accscalar_t(1) / static_cast<accscalar_t>(nframe * dim)
          : accscalar_t(1) / static_cast<accscalar_t>(dim));

  // zero gradients:
  for (int d = item.get_local_id(0); d < dim; d += item.get_local_range(0)) {
    grad_input_k[d] = static_cast<scalar_t>(0);
  }
  sycl::group_barrier(item.get_group());

  // iterate over targets
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = static_cast<int>(target_k[dt]);
    if (target_idx < 0) {
      break;
    }

    // current value for target
    scalar_t input_target_k = input_k[target_idx];

    // compare to all inputs (multithreaded):
    accscalar_t sum = 0;
    for (int d = item.get_local_id(0); d < dim; d += item.get_local_range(0)) {
      // contribute to loss only if not a target
      if (!static_cast<int>(is_target_k[d])) {
        scalar_t z = 1 - input_target_k + input_k[d];
        if (z > 0) {
          sum -= g;
          grad_input_k[d] += g;
        }
      }
    }
    sycl::group_barrier(item.get_group());

    sum = GroupReduceSumWithoutBroadcast<
        accscalar_t,
        MULTILABELMARGIN_SUB_GROUP_SIZE>(item, sum, smem);

    if (item.get_local_id(0) == 0) {
      grad_input_k[target_idx] += static_cast<scalar_t>(sum);
    }
  }

  for (int d = item.get_local_id(0); d < dim; d += item.get_local_range(0)) {
    grad_input_k[d] *= *grad_output_k;
  }
}

void multilabel_margin_loss_kernel(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target) {
  int64_t nframe, dim;
  const int64_t ndims = input.dim();
  multilabel_margin_loss_shape_check(nframe, dim, ndims, input, target);

  if (input.numel() == 0) {
    return;
  }

  auto input_ = input.contiguous();
  auto target_ = target.contiguous();
  auto is_target_ = is_target.contiguous();
  is_target_.resize_as_(target);

  if (input.dim() <= 1) {
    output.resize_({});

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_xpu",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          int64_t local_size = MULTILABELMARGIN_THREADS;
          int slm_sz = sizeof(accscalar_t) * local_size;
          sycl_kernel_submit<multilabel_margin_loss_forward_kernel_impl<
              scalar_t,
              accscalar_t>>(
              local_size,
              local_size,
              getCurrentSYCLQueue(),
              slm_sz,
              output.mutable_data_ptr<scalar_t>(),
              input_.const_data_ptr<scalar_t>(),
              target_.const_data_ptr<int64_t>(),
              is_target_.mutable_data_ptr<scalar_t>(),
              1,
              dim,
              reduction == at::Reduction::Mean);
        });
  } else if (input.dim() == 2) {
    if (reduction != at::Reduction::None) {
      auto output_tmp = at::empty({input_.size(0)}, input_.options());
      output.resize_({});
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "multilabel_margin_loss_xpu",
          [&] {
            using accscalar_t = acc_type_device<scalar_t, kXPU>;
            int64_t local_size = MULTILABELMARGIN_THREADS;
            int slm_sz = sizeof(accscalar_t) * local_size;
            sycl_kernel_submit<multilabel_margin_loss_forward_kernel_impl<
                scalar_t,
                accscalar_t>>(
                input.size(0) * local_size,
                local_size,
                getCurrentSYCLQueue(),
                slm_sz,
                output_tmp.mutable_data_ptr<scalar_t>(),
                input_.const_data_ptr<scalar_t>(),
                target_.const_data_ptr<int64_t>(),
                is_target_.mutable_data_ptr<scalar_t>(),
                nframe,
                dim,
                reduction == at::Reduction::Mean);
          });
      at::sum_out(
          output,
          output_tmp,
          at::IntArrayRef(std::vector<int64_t>{}),
          false,
          output.scalar_type());
    } else {
      output.resize_({input.size(0)});
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "multilabel_margin_loss_xpu",
          [&] {
            using accscalar_t = acc_type_device<scalar_t, kXPU>;
            int64_t local_size = MULTILABELMARGIN_THREADS;
            int slm_sz = sizeof(accscalar_t) * local_size;
            sycl_kernel_submit<multilabel_margin_loss_forward_kernel_impl<
                scalar_t,
                accscalar_t>>(
                input.size(0) * local_size,
                local_size,
                getCurrentSYCLQueue(),
                slm_sz,
                output.mutable_data_ptr<scalar_t>(),
                input_.const_data_ptr<scalar_t>(),
                target_.const_data_ptr<int64_t>(),
                is_target_.mutable_data_ptr<scalar_t>(),
                nframe,
                dim,
                false);
          });
    }

  } else {
    TORCH_CHECK(
        false,
        "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ",
        input.sizes());
  }
}

void multilabel_margin_loss_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  int64_t nframe, dim;
  const int64_t ndims = input.dim();
  multilabel_margin_loss_shape_check(nframe, dim, ndims, input, target);

  if (input.numel() == 0) {
    return;
  }

  auto input_ = input.contiguous();
  auto target_ = target.contiguous();
  auto is_target_ = is_target.contiguous();
  auto grad_output_ = grad_output.contiguous();
  grad_input.resize_as_(input_);

  if (grad_input.dim() <= 1) {
    int target_size = target_.dim() == 0 ? 1 : target_.size(0);
    TORCH_CHECK(
        (target_.numel() != 0) && (target_.dim() <= 1) && (target_size == dim),
        "inconsistent target size");
    TORCH_CHECK(
        target_.sizes() == is_target_.sizes(), "inconsistent is_target size");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_backward_kernel",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          int64_t local_size = MULTILABELMARGIN_THREADS;
          int slm_sz = sizeof(accscalar_t) * local_size;
          sycl_kernel_submit<multilabel_margin_loss_backward_kernel_impl<
              scalar_t,
              accscalar_t>>(
              local_size,
              local_size,
              getCurrentSYCLQueue(),
              slm_sz,
              grad_input.mutable_data_ptr<scalar_t>(),
              grad_output_.const_data_ptr<scalar_t>(),
              input_.const_data_ptr<scalar_t>(),
              target_.const_data_ptr<int64_t>(),
              is_target_.const_data_ptr<scalar_t>(),
              1,
              dim,
              reduction == at::Reduction::Mean,
              reduction != at::Reduction::None);
        });
  } else if (grad_input.dim() == 2) {
    TORCH_CHECK(
        (input_.size(1) != 0) && (target_.dim() == 2) &&
            (target_.size(0) == nframe) && (target_.size(1) == dim),
        "inconsistent target size");
    TORCH_CHECK(
        target_.sizes() == is_target_.sizes(), "inconsistent is_target size");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_backward_kernel",
        [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;
          int64_t local_size = MULTILABELMARGIN_THREADS;
          int slm_sz = sizeof(accscalar_t) * local_size;
          sycl_kernel_submit<multilabel_margin_loss_backward_kernel_impl<
              scalar_t,
              accscalar_t>>(
              grad_input.size(0) * local_size,
              local_size,
              getCurrentSYCLQueue(),
              slm_sz,
              grad_input.mutable_data_ptr<scalar_t>(),
              grad_output_.const_data_ptr<scalar_t>(),
              input_.const_data_ptr<scalar_t>(),
              target_.const_data_ptr<int64_t>(),
              is_target_.const_data_ptr<scalar_t>(),
              grad_input.size(0),
              grad_input.size(1),
              reduction == at::Reduction::Mean,
              reduction != at::Reduction::None);
        });
  } else {
    TORCH_CHECK(
        false,
        "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ",
        grad_input.sizes());
  }
}

} // namespace at::native::xpu
