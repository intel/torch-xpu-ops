/*
 * Copyright 2020-2025 Intel Corporation
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
#include <ATen/native/Resize.h>
#include <ATen/ops/sum.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/MultiMarginLossKernels.h>

namespace at::native::xpu {

using namespace at::xpu;

void multi_margin_loss_shape_check(
    int64_t& nframe,
    int64_t& dim,
    const int64_t& ndims,
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight) {
  TORCH_CHECK(
      (ndims == 2 && input.size(1) != 0) ||
          (ndims == 1 && input.size(0) != 0) || ndims == 0,
      "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
      input.sizes());

  if (ndims <= 1) {
    nframe = 1;
    dim = ndims == 0 ? 1 : input.size(0);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
  }

  TORCH_CHECK(
      target.dim() <= 1 && target.numel() == nframe,
      "inconsistent target size, expected ",
      nframe,
      " but got ",
      target.sizes());
  if (weight && weight->defined()) {
    TORCH_CHECK(
        weight->dim() <= 1 && weight->numel() == dim,
        "inconsistent weight size, expected ",
        dim,
        " but got ",
        weight->sizes());
  }
}

template <int P, typename scalar_t, typename accscalar_t>
struct MultiMarginLossForwardKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    int k = item.get_group(0);
    const scalar_t* input_k = input_ + k * dim_;
    scalar_t* output_k = output_ + k;
    int target_k = static_cast<int>(target_[k]);
    SYCL_KERNEL_ASSERT(
        target_k >= 0 && target_k < dim_ && "target index is out of bounds");
    scalar_t input_target_k = input_k[target_k];
    int i_start = item.get_local_linear_id();
    int i_end = dim_;
    int i_step = item.get_local_range(0);

    smem_[item.get_local_linear_id()] = 0;
    for (int i = i_start; i < i_end; i += i_step) {
      scalar_t z = margin_ - input_target_k + input_k[i];
      if (i == target_k) {
        continue;
      }

      if (z > 0) {
        scalar_t h = (P == 1) ? z : z * z;
        if (weights_) {
          h *= weights_[target_k];
        }
        smem_[item.get_local_linear_id()] += h;
      }
    }
    sycl::group_barrier(item.get_group());

    // reduce
    if (item.get_local_linear_id() == 0) {
      accscalar_t sum = 0;
      for (int i = 0; i < item.get_local_range(0); i++)
        sum += smem_[i];

      const int denom = sizeAverage_ ? nframe_ * dim_ : dim_;
      *output_k = static_cast<scalar_t>(sum / denom);
    }
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<accscalar_t>(smem_size_, cgh);
  }
  MultiMarginLossForwardKernelFunctor(
      scalar_t* output,
      const scalar_t* input,
      const int64_t* target,
      const scalar_t* weights,
      int nframe,
      int dim,
      bool sizeAverage,
      scalar_t margin,
      int64_t smem_size)
      : output_(output),
        input_(input),
        target_(target),
        weights_(weights),
        nframe_(nframe),
        dim_(dim),
        sizeAverage_(sizeAverage),
        margin_(margin),
        smem_size_(smem_size) {}

 private:
  scalar_t* output_;
  const scalar_t* input_;
  const int64_t* target_;
  const scalar_t* weights_;
  int nframe_;
  int dim_;
  bool sizeAverage_;
  scalar_t margin_;
  int64_t smem_size_;
  sycl_local_acc_t<accscalar_t> smem_;
};

template <int P, typename scalar_t, typename accscalar_t>
struct MultiMarginLossBackwardKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    int k = item.get_group(0);
    const scalar_t* input_k = input_ + k * dim_;
    scalar_t* gradInput_k = gradInput_ + k * dim_;
    int target_k = static_cast<int>(target_[k]);
    scalar_t input_target_k = input_k[target_k];

    const scalar_t* gradOutput_k = gradOutput_;
    if (!reduce_) {
      gradOutput_k += k;
    }
    const int denom = sizeAverage_ && reduce_ ? nframe_ * dim_ : dim_;
    const accscalar_t g = accscalar_t(1) / static_cast<accscalar_t>(denom);
    int i_start = item.get_local_linear_id();
    int i_end = dim_;
    int i_step = item.get_local_range(0);

    smem_[item.get_local_linear_id()] = 0;
    for (int i = i_start; i < i_end; i += i_step) {
      scalar_t z = margin_ - input_target_k + input_k[i];
      if (i == target_k) {
        continue;
      }

      if (z > 0) {
        accscalar_t h = (P == 1) ? g : 2 * g * z;
        if (weights_) {
          h *= weights_[target_k];
        }

        smem_[item.get_local_linear_id()] -= static_cast<scalar_t>(h);
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
        gradInput_target_k += smem_[i];
      }

      gradInput_k[target_k] = static_cast<scalar_t>(gradInput_target_k);
    }
    for (int i = i_start; i < i_end; i += i_step) {
      gradInput_k[i] *= *gradOutput_k;
    }
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem_ = sycl_local_acc_t<accscalar_t>(smem_size_, cgh);
  }
  MultiMarginLossBackwardKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* gradOutput,
      const scalar_t* input,
      const int64_t* target,
      const scalar_t* weights,
      int nframe,
      int dim,
      bool sizeAverage,
      scalar_t margin,
      bool reduce,
      int64_t smem_size)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        input_(input),
        target_(target),
        weights_(weights),
        nframe_(nframe),
        dim_(dim),
        sizeAverage_(sizeAverage),
        margin_(margin),
        reduce_(reduce),
        smem_size_(smem_size) {}

 private:
  scalar_t* gradInput_;
  const scalar_t* gradOutput_;
  const scalar_t* input_;
  const int64_t* target_;
  const scalar_t* weights_;
  int nframe_;
  int dim_;
  bool sizeAverage_;
  scalar_t margin_;
  bool reduce_;
  int64_t smem_size_;
  sycl_local_acc_t<accscalar_t> smem_;
};

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
            using KernelClass =
                MultiMarginLossForwardKernelFunctor<1, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
            auto kfn = KernelClass(
                out.mutable_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                target.const_data_ptr<int64_t>(),
                weights.defined() ? weights.const_data_ptr<scalar_t>()
                                  : nullptr,
                1,
                input.dim() < 1 ? input.numel() : input.sizes()[0],
                reduction == at::Reduction::Mean,
                margin,
                local_size);
            sycl_kernel_submit(
                local_size, local_size, getCurrentSYCLQueue(), kfn);
          } else if (p == 2) {
            using KernelClass =
                MultiMarginLossForwardKernelFunctor<2, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
            auto kfn = KernelClass(
                out.mutable_data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                target.const_data_ptr<int64_t>(),
                weights.defined() ? weights.const_data_ptr<scalar_t>()
                                  : nullptr,
                1,
                input.dim() < 1 ? input.numel() : input.sizes()[0],
                reduction == at::Reduction::Mean,
                margin,
                local_size);
            sycl_kernel_submit(
                local_size, local_size, getCurrentSYCLQueue(), kfn);
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
              using KernelClass =
                  MultiMarginLossForwardKernelFunctor<1, scalar_t, accscalar_t>;
              int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
              auto kfn = KernelClass(
                  out.mutable_data_ptr<scalar_t>(),
                  input.const_data_ptr<scalar_t>(),
                  target.const_data_ptr<int64_t>(),
                  weights.defined() ? weights.const_data_ptr<scalar_t>()
                                    : nullptr,
                  nframe,
                  in_sizes[1],
                  false,
                  margin,
                  local_size);
              sycl_kernel_submit(
                  nframe * local_size, local_size, getCurrentSYCLQueue(), kfn);
            } else if (p == 2) {
              using KernelClass =
                  MultiMarginLossForwardKernelFunctor<2, scalar_t, accscalar_t>;
              int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
              auto kfn = KernelClass(
                  out.mutable_data_ptr<scalar_t>(),
                  input.const_data_ptr<scalar_t>(),
                  target.const_data_ptr<int64_t>(),
                  weights.defined() ? weights.const_data_ptr<scalar_t>()
                                    : nullptr,
                  nframe,
                  in_sizes[1],
                  false,
                  margin,
                  local_size);
              sycl_kernel_submit(
                  nframe * local_size, local_size, getCurrentSYCLQueue(), kfn);
            }
          } else {
            auto tmp_output = at::empty({nframe}, input.options());
            if (p == 1) {
              using KernelClass =
                  MultiMarginLossForwardKernelFunctor<1, scalar_t, accscalar_t>;
              int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
              auto kfn = KernelClass(
                  tmp_output.mutable_data_ptr<scalar_t>(),
                  input.const_data_ptr<scalar_t>(),
                  target.const_data_ptr<int64_t>(),
                  weights.defined() ? weights.const_data_ptr<scalar_t>()
                                    : nullptr,
                  nframe,
                  in_sizes[1],
                  reduction == Reduction::Mean,
                  margin,
                  local_size);
              sycl_kernel_submit(
                  nframe * local_size, local_size, getCurrentSYCLQueue(), kfn);

            } else if (p == 2) {
              using KernelClass =
                  MultiMarginLossForwardKernelFunctor<2, scalar_t, accscalar_t>;
              int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
              auto kfn = KernelClass(
                  tmp_output.mutable_data_ptr<scalar_t>(),
                  input.const_data_ptr<scalar_t>(),
                  target.const_data_ptr<int64_t>(),
                  weights.defined() ? weights.const_data_ptr<scalar_t>()
                                    : nullptr,
                  nframe,
                  in_sizes[1],
                  reduction == Reduction::Mean,
                  margin,
                  local_size);
              sycl_kernel_submit(
                  nframe * local_size, local_size, getCurrentSYCLQueue(), kfn);
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
            using KernelClass =
                MultiMarginLossBackwardKernelFunctor<1, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
            auto kfn = KernelClass(
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
                reduction != at::Reduction::None,
                local_size);
            sycl_kernel_submit(
                local_size, local_size, getCurrentSYCLQueue(), kfn);

          } else if (p == 2) {
            using KernelClass =
                MultiMarginLossBackwardKernelFunctor<2, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
            auto kfn = KernelClass(
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
                reduction != at::Reduction::None,
                local_size);
            sycl_kernel_submit(
                local_size, local_size, getCurrentSYCLQueue(), kfn);
          }
        } else {
          auto in_sizes = input.sizes();
          TORCH_INTERNAL_ASSERT(in_sizes.size() == 2);
          TORCH_CHECK(
              (in_sizes[1] != 0) && (target.dim() <= 1) &&
                  (target.numel() == nframe),
              "inconsistent target size");

          if (p == 1) {
            using KernelClass =
                MultiMarginLossBackwardKernelFunctor<1, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
            auto kfn = KernelClass(
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
                reduction != at::Reduction::None,
                local_size);
            sycl_kernel_submit(
                in_sizes[0] * local_size,
                local_size,
                getCurrentSYCLQueue(),
                kfn);

          } else if (p == 2) {
            using KernelClass =
                MultiMarginLossBackwardKernelFunctor<2, scalar_t, accscalar_t>;
            int64_t local_size = syclMaxWorkGroupSize<KernelClass>();
            auto kfn = KernelClass(
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
                reduction != at::Reduction::None,
                local_size);
            sycl_kernel_submit(
                in_sizes[0] * local_size,
                local_size,
                getCurrentSYCLQueue(),
                kfn);
          }
        }
      });

  if (!grad_input.is_alias_of(grad_input_)) {
    grad_input_.copy_(grad_input);
  }
  return grad_input_;
}

} // namespace at::native::xpu
