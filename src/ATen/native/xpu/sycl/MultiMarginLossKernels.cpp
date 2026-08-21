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
#include <ATen/ops/sum.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/MultiMarginLossKernels.h>

namespace at::native::xpu {

using namespace at::xpu;

template <int P, typename scalar_t, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void multi_margin_loss_forward_kernel(
    scalar_t* output,
    const scalar_t* input,
    const int64_t* target,
    const scalar_t* weights,
    int nframe,
    int dim,
    bool sizeAverage,
    scalar_t margin) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int k = item.get_group(0);
  const scalar_t* input_k = input + k * dim;
  scalar_t* output_k = output + k;
  int target_k = static_cast<int>(target[k]);
  SYCL_KERNEL_ASSERT(
      target_k >= 0 && target_k < dim && "target index is out of bounds");
  scalar_t input_target_k = input_k[target_k];
  int i_start = item.get_local_linear_id();
  int i_end = dim;
  int i_step = item.get_local_range(0);

  char* lsm = (char*)syclexp::get_work_group_scratch_memory();
  auto smem = reinterpret_cast<accscalar_t*>(lsm);
  smem[item.get_local_linear_id()] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    scalar_t z = margin - input_target_k + input_k[i];
    if (i == target_k) {
      continue;
    }

    if (z > 0) {
      scalar_t h = (P == 1) ? z : z * z;
      if (weights) {
        h *= weights[target_k];
      }
      smem[item.get_local_linear_id()] += h;
    }
  }
  sycl::group_barrier(item.get_group());

  // reduce
  if (item.get_local_linear_id() == 0) {
    accscalar_t sum = 0;
    for (int i = 0; i < item.get_local_range(0); i++)
      sum += smem[i];

    const int denom = sizeAverage ? nframe * dim : dim;
    *output_k = static_cast<scalar_t>(sum / denom);
  }
}

template <int P, typename scalar_t, typename accscalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void multi_margin_loss_backward_kernel(
    scalar_t* gradInput,
    const scalar_t* gradOutput,
    const scalar_t* input,
    const int64_t* target,
    const scalar_t* weights,
    int nframe,
    int dim,
    bool sizeAverage,
    scalar_t margin,
    bool reduce) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int k = item.get_group(0);
  const scalar_t* input_k = input + k * dim;
  scalar_t* gradInput_k = gradInput + k * dim;
  int target_k = static_cast<int>(target[k]);
  scalar_t input_target_k = input_k[target_k];

  const scalar_t* gradOutput_k = gradOutput;
  if (!reduce) {
    gradOutput_k += k;
  }
  const int denom = sizeAverage && reduce ? nframe * dim : dim;
  const accscalar_t g = accscalar_t(1) / static_cast<accscalar_t>(denom);
  int i_start = item.get_local_linear_id();
  int i_end = dim;
  int i_step = item.get_local_range(0);

  char* lsm = (char*)syclexp::get_work_group_scratch_memory();
  auto smem = reinterpret_cast<accscalar_t*>(lsm);
  smem[item.get_local_linear_id()] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    scalar_t z = margin - input_target_k + input_k[i];
    if (i == target_k) {
      continue;
    }

    if (z > 0) {
      accscalar_t h = (P == 1) ? g : 2 * g * z;
      if (weights) {
        h *= weights[target_k];
      }

      smem[item.get_local_linear_id()] -= static_cast<scalar_t>(h);
      gradInput_k[i] = static_cast<scalar_t>(h);
    } else {
      gradInput_k[i] = static_cast<scalar_t>(0);
    }
  }
  sycl::group_barrier(item.get_group());

  // reduce
  if (item.get_local_linear_id() == 0) {
    accscalar_t gradInput_target_k = 0;

    for (int i = 0; i < item.get_local_range(0); i++) {
      gradInput_target_k += smem[i];
    }

    gradInput_k[target_k] = static_cast<scalar_t>(gradInput_target_k);
  }
  for (int i = i_start; i < i_end; i += i_step) {
    gradInput_k[i] *= *gradOutput_k;
  }
}

Tensor& multi_margin_loss_kernel(
    const Tensor& input_,
    const Tensor& target_,
    const Scalar& p_,
    const Scalar& margin_,
    const std::optional<Tensor>& weights_,
    int64_t reduction,
    Tensor& out_) {
  auto p = p_.toLong();
  int64_t nframe, dim;
  const auto ndims = input_.dim();
  TORCH_CHECK(
      p == 1 || p == 2,
      "multi_margin_loss: Invalid p, expected 1 or 2 but got ",
      p);

  multi_margin_loss_shape_check(nframe, dim, ndims, input_, target_, weights_);

  // produce a scalar output for 1d input
  if (reduction == Reduction::None && target_.dim() > 0) {
    resize_output(out_, {nframe});
  } else {
    resize_output(out_, {});
  }
  if (input_.numel() == 0) {
    return out_;
  }

  auto input = input_.contiguous();
  auto target = target_.contiguous();
  Tensor weights;
  if (weights_ && weights_->defined()) {
    weights = weights_->contiguous();
  }
  auto out =
      (out_.is_contiguous() ? out_ : at::empty(out_.sizes(), input.options()));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "multi_margin_loss_xpu", [&] {
        const scalar_t margin = margin_.to<scalar_t>();
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        if (input.dim() <= 1) {
          TORCH_CHECK(
              target.dim() <= 1 && target.numel() == nframe,
              "inconsistent target size");

          if (p == 1) {
            constexpr auto kernelFunc =
                multi_margin_loss_forward_kernel<1, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
            sycl_kernel_submit<kernelFunc>(
                local_size,
                local_size,
                getCurrentSYCLQueue(),
                local_size * sizeof(accscalar_t),
                out.mutable_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                target.const_data_ptr<int64_t>(),
                weights.defined() ? weights.const_data_ptr<scalar_t>()
                                  : nullptr,
                1,
                input.dim() < 1 ? input.numel() : input.sizes()[0],
                reduction == at::Reduction::Mean,
                margin);
          } else if (p == 2) {
            constexpr auto kernelFunc =
                multi_margin_loss_forward_kernel<2, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
            sycl_kernel_submit<kernelFunc>(
                local_size,
                local_size,
                getCurrentSYCLQueue(),
                local_size * sizeof(accscalar_t),
                out.mutable_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                target.const_data_ptr<int64_t>(),
                weights.defined() ? weights.const_data_ptr<scalar_t>()
                                  : nullptr,
                1,
                input.dim() < 1 ? input.numel() : input.sizes()[0],
                reduction == at::Reduction::Mean,
                margin);
          }
        } else {
          auto in_sizes = input.sizes();
          TORCH_INTERNAL_ASSERT(in_sizes.size() == 2);
          // allow zero-dim target for 2D input.
          TORCH_CHECK(
              in_sizes[1] != 0 && target.dim() <= 1 && target.numel() == nframe,
              "inconsistent target size");

          if (reduction == at::Reduction::None) {
            if (p == 1) {
              constexpr auto kernelFunc =
                  multi_margin_loss_forward_kernel<1, scalar_t, accscalar_t>;
              int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
              sycl_kernel_submit<kernelFunc>(
                  nframe * local_size,
                  local_size,
                  getCurrentSYCLQueue(),
                  local_size * sizeof(accscalar_t),
                  out.mutable_data_ptr<scalar_t>(),
                  input.const_data_ptr<scalar_t>(),
                  target.const_data_ptr<int64_t>(),
                  weights.defined() ? weights.const_data_ptr<scalar_t>()
                                    : nullptr,
                  nframe,
                  in_sizes[1],
                  false,
                  margin);
            } else if (p == 2) {
              constexpr auto kernelFunc =
                  multi_margin_loss_forward_kernel<2, scalar_t, accscalar_t>;
              int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
              sycl_kernel_submit<kernelFunc>(
                  nframe * local_size,
                  local_size,
                  getCurrentSYCLQueue(),
                  local_size * sizeof(accscalar_t),
                  out.mutable_data_ptr<scalar_t>(),
                  input.const_data_ptr<scalar_t>(),
                  target.const_data_ptr<int64_t>(),
                  weights.defined() ? weights.const_data_ptr<scalar_t>()
                                    : nullptr,
                  nframe,
                  in_sizes[1],
                  false,
                  margin);
            }
          } else {
            auto tmp_output = at::empty({nframe}, input.options());
            if (p == 1) {
              constexpr auto kernelFunc =
                  multi_margin_loss_forward_kernel<1, scalar_t, accscalar_t>;
              int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
              sycl_kernel_submit<kernelFunc>(
                  nframe * local_size,
                  local_size,
                  getCurrentSYCLQueue(),
                  local_size * sizeof(accscalar_t),
                  tmp_output.mutable_data_ptr<scalar_t>(),
                  input.const_data_ptr<scalar_t>(),
                  target.const_data_ptr<int64_t>(),
                  weights.defined() ? weights.const_data_ptr<scalar_t>()
                                    : nullptr,
                  nframe,
                  in_sizes[1],
                  reduction == Reduction::Mean,
                  margin);
            } else if (p == 2) {
              constexpr auto kernelFunc =
                  multi_margin_loss_forward_kernel<2, scalar_t, accscalar_t>;
              int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
              sycl_kernel_submit<kernelFunc>(
                  nframe * local_size,
                  local_size,
                  getCurrentSYCLQueue(),
                  local_size * sizeof(accscalar_t),
                  tmp_output.mutable_data_ptr<scalar_t>(),
                  input.const_data_ptr<scalar_t>(),
                  target.const_data_ptr<int64_t>(),
                  weights.defined() ? weights.const_data_ptr<scalar_t>()
                                    : nullptr,
                  nframe,
                  in_sizes[1],
                  reduction == Reduction::Mean,
                  margin);
            }
            at::sum_out(out, tmp_output, IntArrayRef{});
          }
        }
      });
  if (!out.is_alias_of(out_)) {
    out_.copy_(out);
  }
  return out_;
}

Tensor& multi_margin_loss_backward_kernel(
    const Tensor& grad_output_,
    const Tensor& input_,
    const Tensor& target_,
    const Scalar& p_,
    const Scalar& margin_,
    const std::optional<Tensor>& weights_,
    int64_t reduction,
    Tensor& grad_input_) {
  auto p = p_.toLong();
  int64_t nframe, dim;
  const auto ndims = input_.dim();

  TORCH_CHECK(
      p == 1 || p == 2,
      "multi_margin_loss_backward: Invalid p, expected 1 or 2 but got ",
      p);

  multi_margin_loss_shape_check(nframe, dim, ndims, input_, target_, weights_);
  resize_output(grad_input_, input_.sizes());

  if (input_.numel() == 0) {
    return grad_input_;
  }

  auto input = input_.contiguous();
  auto grad_input =
      (grad_input_.is_contiguous()
           ? grad_input_
           : at::empty(grad_input_.sizes(), input.options()));
  auto grad_output = grad_output_.contiguous();
  auto target = target_.contiguous();
  Tensor weights;
  if (weights_ && weights_->defined()) {
    weights = weights_->contiguous();
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "multi_margin_loss_backward_xpu",
      [&] {
        const scalar_t margin = margin_.to<scalar_t>();
        using accscalar_t = acc_type_device<scalar_t, kXPU>;

        if (input.dim() <= 1) {
          if (p == 1) {
            constexpr auto kernelFunc =
                multi_margin_loss_backward_kernel<1, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
            sycl_kernel_submit<kernelFunc>(
                local_size,
                local_size,
                getCurrentSYCLQueue(),
                local_size * sizeof(accscalar_t),
                grad_input.mutable_data_ptr<scalar_t>(),
                grad_output.const_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                target.const_data_ptr<int64_t>(),
                weights.defined() ? weights.const_data_ptr<scalar_t>()
                                  : nullptr,
                1,
                input.dim() == 0 ? 1 : input.sizes()[0],
                reduction == at::Reduction::Mean,
                margin,
                reduction != at::Reduction::None);
          } else if (p == 2) {
            constexpr auto kernelFunc =
                multi_margin_loss_backward_kernel<2, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
            sycl_kernel_submit<kernelFunc>(
                local_size,
                local_size,
                getCurrentSYCLQueue(),
                local_size * sizeof(accscalar_t),
                grad_input.mutable_data_ptr<scalar_t>(),
                grad_output.const_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                target.const_data_ptr<int64_t>(),
                weights.defined() ? weights.const_data_ptr<scalar_t>()
                                  : nullptr,
                1,
                input.dim() == 0 ? 1 : input.sizes()[0],
                reduction == at::Reduction::Mean,
                margin,
                reduction != at::Reduction::None);
          }
        } else {
          auto in_sizes = input.sizes();
          TORCH_INTERNAL_ASSERT(in_sizes.size() == 2);
          TORCH_CHECK(
              (in_sizes[1] != 0) && (target.dim() <= 1) &&
                  (target.numel() == nframe),
              "inconsistent target size");

          if (p == 1) {
            constexpr auto kernelFunc =
                multi_margin_loss_backward_kernel<1, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
            sycl_kernel_submit<kernelFunc>(
                in_sizes[0] * local_size,
                local_size,
                getCurrentSYCLQueue(),
                local_size * sizeof(accscalar_t),
                grad_input.mutable_data_ptr<scalar_t>(),
                grad_output.const_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                target.const_data_ptr<int64_t>(),
                weights.defined() ? weights.const_data_ptr<scalar_t>()
                                  : nullptr,
                nframe,
                in_sizes[1],
                reduction == at::Reduction::Mean,
                margin,
                reduction != at::Reduction::None);
          } else if (p == 2) {
            constexpr auto kernelFunc =
                multi_margin_loss_backward_kernel<2, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<kernelFunc>();
            sycl_kernel_submit<kernelFunc>(
                in_sizes[0] * local_size,
                local_size,
                getCurrentSYCLQueue(),
                local_size * sizeof(accscalar_t),
                grad_input.mutable_data_ptr<scalar_t>(),
                grad_output.const_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                target.const_data_ptr<int64_t>(),
                weights.defined() ? weights.const_data_ptr<scalar_t>()
                                  : nullptr,
                nframe,
                in_sizes[1],
                reduction == at::Reduction::Mean,
                margin,
                reduction != at::Reduction::None);
          }
        }
      });

  if (!grad_input.is_alias_of(grad_input_)) {
    grad_input_.copy_(grad_input);
  }
  return grad_input_;
}

} // namespace at::native::xpu
