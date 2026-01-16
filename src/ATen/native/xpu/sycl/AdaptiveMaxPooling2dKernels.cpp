/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/AdaptiveMaxPooling2dKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename index_t>
struct AdaptiveMaxPool2dKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);

    do {
      if (desc.glb_problem >= cfg_.problem_)
        break;

      int64_t o_lid = desc.glb_problem;
      int64_t ob = o_lid / ostrideB_;
      int64_t op = (o_lid / ostrideP_) % sizeP_;
      int64_t oh = (o_lid / ostrideH_) % osizeH_;
      int64_t ow = o_lid % osizeW_;
      int64_t o_off = o_lid;

      int64_t istartH = start_index(oh, osizeH_, isizeH_);
      int64_t iendH = end_index(oh, osizeH_, isizeH_);
      int64_t istartW = start_index(ow, osizeW_, isizeW_);
      int64_t iendW = end_index(ow, osizeW_, isizeW_);

      scalar_t max = at::numeric_limits<scalar_t>::lower_bound();
      index_t argmax;
      int64_t i_bp_off = ob * istrideB_ + op * istrideP_;
      for (int64_t ih = istartH; ih < iendH; ih++) {
        for (int64_t iw = istartW; iw < iendW; iw++) {
          int64_t i_hw_off = ih * istrideH_ + iw * istrideW_;
          int64_t i_hw_id = ih * isizeW_ + iw;
          scalar_t val = input_[i_bp_off + i_hw_off];
          if ((val > max) || at::_isnan(val)) {
            max = val;
            argmax = i_hw_id;
          }
        }
      }
      output_[o_off] = max;
      indices_[o_off] = argmax;
    } while (cfg_.next(item, desc));
  }

  AdaptiveMaxPool2dKernelFunctor(
      const scalar_t* input,
      scalar_t* output,
      index_t* indices,
      int64_t sizeP,
      int64_t isizeH,
      int64_t isizeW,
      int64_t osizeH,
      int64_t osizeW,
      int64_t istrideB,
      int64_t istrideP,
      int64_t istrideH,
      int64_t istrideW,
      BatchKernelConfig cfg)
      : input_(input),
        output_(output),
        indices_(indices),
        sizeP_(sizeP),
        isizeH_(isizeH),
        isizeW_(isizeW),
        osizeH_(osizeH),
        osizeW_(osizeW),
        istrideB_(istrideB),
        istrideP_(istrideP),
        istrideH_(istrideH),
        istrideW_(istrideW),
        cfg_(cfg) {
    // assume output tensor is in contiguous format
    ostrideH_ = osizeW;
    ostrideP_ = osizeW * osizeH;
    ostrideB_ = ostrideP_ * sizeP;
  }

 private:
  const scalar_t* input_;
  scalar_t* output_;
  index_t* indices_;
  int64_t sizeP_;
  int64_t isizeH_;
  int64_t isizeW_;
  int64_t osizeH_;
  int64_t osizeW_;
  int64_t istrideB_;
  int64_t istrideP_;
  int64_t istrideH_;
  int64_t istrideW_;
  BatchKernelConfig cfg_;
  int64_t ostrideB_;
  int64_t ostrideP_;
  int64_t ostrideH_;
};

template <typename scalar_t, typename index_t>
void launch_adaptive_max_pool2d_kernel(
    const scalar_t* input,
    scalar_t* output,
    index_t* indices,
    int64_t batch,
    int64_t plane,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeH,
    int64_t osizeW,
    int64_t istrideB,
    int64_t istrideP,
    int64_t istrideH,
    int64_t istrideW) {
  using KernelClass = AdaptiveMaxPool2dKernelFunctor<scalar_t, index_t>;

  int64_t output_size = batch * plane * osizeH * osizeW;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      1, output_size, 1, 1, true, BatchKernelConfig::Policy::pAdaptive);

  cfg.build<KernelClass>();

  auto kfn = KernelClass(
      input,
      output,
      indices,
      plane,
      isizeH,
      isizeW,
      osizeH,
      osizeW,
      istrideB,
      istrideP,
      istrideH,
      istrideW,
      cfg);

  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);
}

void adaptive_max_pool2d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    const Tensor& output,
    const Tensor& indices) {
  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  const at::Tensor output_c = output.is_contiguous()
      ? output
      : at::empty(output.sizes(), output.options());
  const at::Tensor indices_c = indices.is_contiguous()
      ? indices
      : at::empty(indices.sizes(), indices.options());

  if (input.ndimension() == 3) {
    int64_t plane = input.size(0);
    int64_t isizeH = input.size(1);
    int64_t isizeW = input.size(2);

    int64_t istrideP = input.stride(0);
    int64_t istrideH = input.stride(1);
    int64_t istrideW = input.stride(2);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "adaptive_max_pool2d_xpu", [&] {
          const scalar_t* input_data = input.const_data_ptr<scalar_t>();
          scalar_t* output_data = output_c.mutable_data_ptr<scalar_t>();
          int64_t* indices_data = indices_c.mutable_data_ptr<int64_t>();

          launch_adaptive_max_pool2d_kernel<scalar_t, int64_t>(
              input_data,
              output_data,
              indices_data,
              1,
              plane,
              isizeH,
              isizeW,
              osizeH,
              osizeW,
              istrideP,
              istrideP,
              istrideH,
              istrideW);
        });
  } else {
    int64_t batch = input.size(0);
    int64_t plane = input.size(1);
    int64_t isizeH = input.size(2);
    int64_t isizeW = input.size(3);

    int64_t istrideB = input.stride(0);
    int64_t istrideP = input.stride(1);
    int64_t istrideH = input.stride(2);
    int64_t istrideW = input.stride(3);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "adaptive_max_pool2d_xpu", [&] {
          const scalar_t* input_data = input.const_data_ptr<scalar_t>();
          scalar_t* output_data = output_c.mutable_data_ptr<scalar_t>();
          int64_t* indices_data = indices_c.mutable_data_ptr<int64_t>();

          launch_adaptive_max_pool2d_kernel<scalar_t, int64_t>(
              input_data,
              output_data,
              indices_data,
              batch,
              plane,
              isizeH,
              isizeW,
              osizeH,
              osizeW,
              istrideB,
              istrideP,
              istrideH,
              istrideW);
        });

    if (!output.is_contiguous()) {
      output.copy_(output_c);
    }
    if (!indices.is_contiguous()) {
      indices.copy_(indices_c);
    }
  }
}

template <typename scalar_t, typename index_t>
struct AdaptiveMaxPool2dBackwardKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);

    do {
      if (desc.glb_problem >= cfg_.problem_)
        break;

      int64_t o_lid = desc.glb_problem;
      int64_t ob = o_lid / ostrideB_;
      int64_t op = (o_lid / ostrideP_) % sizeP_;
      int64_t o_off = o_lid;
      int64_t i_off = ob * istrideB_ + op * istrideP_;

      index_t idx = indices_[o_off];
      auto target = sycl_global_ptr<scalar_t>(grad_input_ + i_off + idx);
      atomicAdd(target, grad_output_[o_off]);
    } while (cfg_.next(item, desc));
  }

  AdaptiveMaxPool2dBackwardKernelFunctor(
      const scalar_t* grad_output,
      const index_t* indices,
      scalar_t* grad_input,
      int64_t istrideB,
      int64_t istrideP,
      int64_t ostrideB,
      int64_t ostrideP,
      int64_t sizeP,
      BatchKernelConfig cfg)
      : grad_output_(grad_output),
        indices_(indices),
        grad_input_(grad_input),
        istrideB_(istrideB),
        istrideP_(istrideP),
        ostrideB_(ostrideB),
        ostrideP_(ostrideP),
        sizeP_(sizeP),
        cfg_(cfg) {}

 private:
  const scalar_t* grad_output_;
  const index_t* indices_;
  scalar_t* grad_input_;
  int64_t istrideB_;
  int64_t istrideP_;
  int64_t ostrideB_;
  int64_t ostrideP_;
  int64_t sizeP_;
  BatchKernelConfig cfg_;
};

template <typename scalar_t, typename index_t>
void launch_adaptive_max_pool2d_backward_kernel(
    const scalar_t* grad_output,
    const index_t* indices,
    scalar_t* grad_input,
    int64_t osize,
    int64_t istrideB,
    int64_t istrideP,
    int64_t ostrideB,
    int64_t ostrideP,
    int64_t sizeP) {
  using KernelClass = AdaptiveMaxPool2dBackwardKernelFunctor<scalar_t, index_t>;

  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      1, osize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive);

  cfg.build<KernelClass>();

  auto kfn = KernelClass(
      grad_output,
      indices,
      grad_input,
      istrideB,
      istrideP,
      ostrideB,
      ostrideP,
      sizeP,
      cfg);

  sycl_kernel_submit(
      cfg.global_size(), cfg.group_size(), getCurrentSYCLQueue(), kfn);
}

void adaptive_max_pool2d_backward_kernel(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& grad_input) {
  globalContext().alertNotDeterministic("adaptive_max_pool2d_backward_xpu");

  const at::Tensor grad_output_ = grad_output.contiguous();
  const at::Tensor indices_ = indices.contiguous();
  const at::Tensor grad_input_c = grad_input.is_contiguous()
      ? grad_input
      : at::empty(grad_input.sizes(), grad_input.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "adaptive_max_pool2d_backward_xpu",
      [&] {
        scalar_t* grad_input_data = grad_input_c.mutable_data_ptr<scalar_t>();
        const scalar_t* grad_output_data =
            grad_output_.const_data_ptr<scalar_t>();
        const int64_t* indices_data = indices_.const_data_ptr<int64_t>();

        grad_input_c.zero_();

        int64_t istrideB;
        int64_t istrideP;
        int64_t ostrideB;
        int64_t ostrideP;
        int64_t sizeP;
        if (input.ndimension() == 3) {
          istrideP = input.size(1) * input.size(2);
          istrideB = istrideP * input.size(0);
          ostrideP = grad_output_.size(1) * grad_output_.size(2);
          ostrideB = ostrideP * grad_output_.size(0);
          sizeP = grad_output_.size(0);
        } else {
          istrideP = input.size(2) * input.size(3);
          istrideB = istrideP * input.size(1);
          ostrideP = grad_output_.size(2) * grad_output_.size(3);
          ostrideB = ostrideP * grad_output_.size(1);
          sizeP = grad_output_.size(1);
        }

        launch_adaptive_max_pool2d_backward_kernel(
            grad_output_data,
            indices_data,
            grad_input_data,
            grad_output.numel(),
            istrideB,
            istrideP,
            ostrideB,
            ostrideP,
            sizeP);
      });

  if (!grad_input.is_contiguous()) {
    grad_input.copy_(grad_input_c);
  }
}

} // namespace at::native::xpu
