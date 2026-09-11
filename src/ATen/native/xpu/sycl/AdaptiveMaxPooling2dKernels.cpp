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
#include <ATen/NumericUtils.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/AdaptiveMaxPooling2dKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void adaptive_max_pool2d_kernel_impl(
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
    BatchKernelConfig cfg) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  auto desc = cfg.get_item_desc(item);

  auto ostrideH = osizeW;
  auto ostrideP = osizeW * osizeH;
  auto ostrideB = ostrideP * sizeP;

  do {
    if (desc.glb_problem >= cfg.problem_)
      break;

    int64_t o_lid = desc.glb_problem;
    int64_t ob = o_lid / ostrideB;
    int64_t op = (o_lid / ostrideP) % sizeP;
    int64_t oh = (o_lid / ostrideH) % osizeH;
    int64_t ow = o_lid % osizeW;
    int64_t o_off = o_lid;

    int64_t istartH = start_index(oh, osizeH, isizeH);
    int64_t iendH = end_index(oh, osizeH, isizeH);
    int64_t istartW = start_index(ow, osizeW, isizeW);
    int64_t iendW = end_index(ow, osizeW, isizeW);

    scalar_t max = at::numeric_limits<scalar_t>::lower_bound();
    index_t argmax = istartH * isizeW + istartW;
    int64_t i_bp_off = ob * istrideB + op * istrideP;
    for (int64_t ih = istartH; ih < iendH; ih++) {
      for (int64_t iw = istartW; iw < iendW; iw++) {
        int64_t i_hw_off = ih * istrideH + iw * istrideW;
        int64_t i_hw_id = ih * isizeW + iw;
        scalar_t val = input[i_bp_off + i_hw_off];
        if ((val > max) || at::_isnan(val)) {
          max = val;
          argmax = i_hw_id;
        }
      }
    }
    output[o_off] = max;
    indices[o_off] = argmax;
  } while (cfg.next(item, desc));
}

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
  int64_t output_size = batch * plane * osizeH * osizeW;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<
      adaptive_max_pool2d_kernel_impl<scalar_t, index_t>>(
      1, output_size, 1, 1, true, {BatchKernelConfig::Policy::pAdaptive});

  cfg.build<adaptive_max_pool2d_kernel_impl<scalar_t, index_t>>();

  sycl_kernel_submit<adaptive_max_pool2d_kernel_impl<scalar_t, index_t>>(
      cfg.global_size(),
      cfg.group_size(),
      getCurrentSYCLQueue(),
      0,
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
  }

  if (!output.is_contiguous()) {
    output.copy_(output_c);
  }
  if (!indices.is_contiguous()) {
    indices.copy_(indices_c);
  }
}

template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void adaptive_avg_pool2d_backward_kernel_impl(
    const scalar_t* grad_output,
    const index_t* indices,
    scalar_t* grad_input,
    int64_t istrideB,
    int64_t istrideP,
    int64_t ostrideB,
    int64_t ostrideP,
    int64_t sizeP,
    BatchKernelConfig cfg) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  auto desc = cfg.get_item_desc(item);

  do {
    if (desc.glb_problem >= cfg.problem_)
      break;

    int64_t o_lid = desc.glb_problem;
    int64_t ob = o_lid / ostrideB;
    int64_t op = (o_lid / ostrideP) % sizeP;
    int64_t o_off = o_lid;
    int64_t i_off = ob * istrideB + op * istrideP;

    index_t idx = indices[o_off];
    auto target = sycl_global_ptr<scalar_t>(grad_input + i_off + idx);
    atomicAdd(target, grad_output[o_off]);
  } while (cfg.next(item, desc));
}

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
  BatchKernelConfig cfg = BatchKernelConfig::make_config<
      adaptive_avg_pool2d_backward_kernel_impl<scalar_t, index_t>>(
      1, osize, 1, 1, true, {BatchKernelConfig::Policy::pAdaptive});

  cfg.build<adaptive_avg_pool2d_backward_kernel_impl<scalar_t, index_t>>();

  sycl_kernel_submit<
      adaptive_avg_pool2d_backward_kernel_impl<scalar_t, index_t>>(
      cfg.global_size(),
      cfg.group_size(),
      getCurrentSYCLQueue(),
      0,
      grad_output,
      indices,
      grad_input,
      istrideB,
      istrideP,
      ostrideB,
      ostrideP,
      sizeP,
      cfg);
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

// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
