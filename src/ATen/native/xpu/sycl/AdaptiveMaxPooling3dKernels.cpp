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

#include <ATen/native/xpu/sycl/AdaptiveMaxPooling3dKernels.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void adaptive_max_pool3d_kernel_impl(
    const scalar_t* input_data,
    scalar_t* output_data,
    int64_t* indices_data,
    int isizeT,
    int isizeH,
    int isizeW,
    int osizeT,
    int osizeH,
    int osizeW,
    int64_t istrideD,
    int64_t istrideT,
    int64_t istrideH,
    int64_t istrideW,
    int64_t offsetZ) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  // iterators on output pixels
  int ot, oh, ow;

  int ostartH =
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  int oendH = osizeH;
  int ostepH = item.get_group_range(0) * item.get_local_range(0);
  int ostartW = item.get_local_id(1);
  int oendW = osizeW;
  int ostepW = item.get_local_range(1);

  // select output plane
  int64_t o_plane = item.get_group(1) + offsetZ;
  ot = o_plane % osizeT;
  int d = o_plane / osizeT;

  int istartT = start_index(ot, osizeT, isizeT);
  int iendT = end_index(ot, osizeT, isizeT);
  int kT = iendT - istartT;

  const scalar_t* input_dt = input_data + d * istrideD + istartT * istrideT;

  scalar_t* output_dt = output_data + o_plane * osizeH * osizeW;

  int64_t* indices_dt = indices_data + o_plane * osizeH * osizeW;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    int istartH = start_index(oh, osizeH, isizeH);
    int iendH = end_index(oh, osizeH, isizeH);
    int kH = iendH - istartH;

    for (ow = ostartW; ow < oendW; ow += ostepW) {
      int istartW = start_index(ow, osizeW, isizeW);
      int iendW = end_index(ow, osizeW, isizeW);
      int kW = iendW - istartW;

      // Compute the average pooling from corresponding input pixels
      const scalar_t* ptr_input =
          input_dt + istartH * istrideH + istartW * istrideW;
      scalar_t* ptr_output = output_dt + oh * osizeW + ow;
      int64_t* ptr_ind = indices_dt + oh * osizeW + ow;
      int64_t argmax = istartT * isizeH * isizeW + istartH * isizeW + istartW;
      scalar_t max = at::numeric_limits<scalar_t>::lower_bound(); // -Infinity

      int it, ih, iw;
      for (it = 0; it < kT; ++it) {
        for (ih = 0; ih < kH; ++ih) {
          for (iw = 0; iw < kW; ++iw) {
            scalar_t val = ptr_input[ih * istrideH + iw * istrideW];
            if ((val > max) || at::_isnan(val)) {
              max = val;
              argmax = (it + istartT) * isizeH * isizeW +
                  (ih + istartH) * isizeW + iw + istartW;
            }
          }
        }
        ptr_input += istrideT; // next input frame
      }
      // Update output and argmax
      *ptr_output = max;
      *ptr_ind = argmax;
    }
  }
}

template <typename scalar_t>
void adaptive_max_pool3d_template(
    const scalar_t* input_data,
    scalar_t* output_data,
    int64_t* indices_data,
    int64_t totalZ,
    int isizeT,
    int isizeH,
    int isizeW,
    int osizeT,
    int osizeH,
    int osizeW,
    int64_t istrideD,
    int64_t istrideT,
    int64_t istrideH,
    int64_t istrideW) {
  int64_t offsetZ = 0;
  int width_group_size = 32;
  int height_group_size = 8;
  int height_group_range = std::max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    int width_group_range = totalZ > 65535 ? 65535 : totalZ;
    auto& queue = getCurrentSYCLQueue();
    sycl_kernel_submit<adaptive_max_pool3d_kernel_impl<scalar_t>>(
        sycl::range<2>{
            size_t(height_group_range * height_group_size),
            size_t(width_group_range * width_group_size),
        },
        sycl::range<2>{size_t(height_group_size), size_t(width_group_size)},
        queue,
        0,
        input_data,
        output_data,
        indices_data,
        isizeT,
        isizeH,
        isizeW,
        osizeT,
        osizeH,
        osizeW,
        istrideD,
        istrideT,
        istrideH,
        istrideW,
        offsetZ);
    totalZ -= 65535;
    offsetZ += 65535;
  }
}
void adaptive_max_pool3d_kernel(
    const Tensor& input,
    IntArrayRef output_size,
    const Tensor& output,
    const Tensor& indices) {
  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t istrideD, istrideT, istrideH, istrideW;
  int64_t totalZ;

  const Tensor& input_ = input.ndimension() == 4 ? input : input.contiguous();
  if (input_.ndimension() == 4) {
    sizeD = input_.size(0);
    isizeT = input_.size(1);
    isizeH = input_.size(2);
    isizeW = input_.size(3);

    istrideD = input_.stride(0);
    istrideT = input_.stride(1);
    istrideH = input_.stride(2);
    istrideW = input_.stride(3);

    totalZ = sizeD * osizeT;
  } else {
    int64_t sizeB = input_.size(0);
    sizeD = input_.size(1);
    isizeT = input_.size(2);
    isizeH = input_.size(3);
    isizeW = input_.size(4);

    // In the kernel, the batch and channel dimensions are treated as if they
    // are flattened and istrideD is used as the stride of this flattened dim
    // Handle the edge case where input_.size(1) == 1, where despite passing
    // the contiguity check the stride might not be T * H * W
    istrideD = isizeT * isizeH * isizeW;
    istrideT = input_.stride(2);
    istrideH = input_.stride(3);
    istrideW = input_.stride(4);

    totalZ = sizeB * sizeD * osizeT;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "adaptive_max_pool3d_xpu", [&] {
        const scalar_t* input_data = input_.const_data_ptr<scalar_t>();
        scalar_t* output_data = output.mutable_data_ptr<scalar_t>();
        int64_t* indices_data = indices.mutable_data_ptr<int64_t>();

        adaptive_max_pool3d_template(
            input_data,
            output_data,
            indices_data,
            totalZ,
            isizeT,
            isizeH,
            isizeW,
            osizeT,
            osizeH,
            osizeW,
            istrideD,
            istrideT,
            istrideH,
            istrideW);
      });
}

template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void adaptive_max_pool3d_backward_atomic_kernel(
    scalar_t* gradInput,
    const scalar_t* gradOutput,
    const int64_t* indices,
    int isizeT,
    int isizeH,
    int isizeW,
    int osizeT,
    int osizeH,
    int osizeW,
    int64_t offsetZ) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  // iterators on output pixels
  int oh, ow;

  int ostartH =
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  int oendH = osizeH;
  int ostepH = item.get_group_range(0) * item.get_local_range(0);
  int ostartW = item.get_local_id(1);
  int oendW = osizeW;
  int ostepW = item.get_local_range(1);

  // select output plane
  int64_t o_plane = item.get_group(1) + offsetZ;
  int d = o_plane / osizeT;

  scalar_t* gradInput_dt = gradInput + d * isizeT * isizeH * isizeW;
  const scalar_t* gradOutput_dt = gradOutput + o_plane * osizeH * osizeW;
  const int64_t* indices_dt = indices + o_plane * osizeH * osizeW;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    for (ow = ostartW; ow < oendW; ow += ostepW) {
      // Compute the gradients for the argmax input pixel
      const scalar_t* ptr_gradOutput = gradOutput_dt + oh * osizeW + ow;
      const int64_t* ptr_ind = indices_dt + oh * osizeW + ow;
      scalar_t grad_delta = *ptr_gradOutput;
      int64_t argmax = (*ptr_ind);
      atomicAdd((sycl_global_ptr<scalar_t>)&(gradInput_dt[argmax]), grad_delta);
    }
  }
}

template <typename scalar_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<2>))
void adaptive_max_pool3d_backward_kernel_impl(
    scalar_t* gradInput,
    const scalar_t* gradOutput,
    const int64_t* indices,
    int isizeT,
    int isizeH,
    int isizeW,
    int osizeT,
    int osizeH,
    int osizeW,
    int64_t offsetZ) {
  auto item = syclext::this_work_item::get_nd_item<2>();
  // iterators on output pixels
  int oh, ow;

  int ostartH =
      item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
  int oendH = osizeH;
  int ostepH = item.get_group_range(0) * item.get_local_range(0);
  int ostartW = item.get_local_id(1);
  int oendW = osizeW;
  int ostepW = item.get_local_range(1);

  // select output plane
  int64_t o_plane = item.get_group(1) + offsetZ;
  int d = o_plane / osizeT;

  scalar_t* gradInput_dt = gradInput + d * isizeT * isizeH * isizeW;
  const scalar_t* gradOutput_dt = gradOutput + o_plane * osizeH * osizeW;
  const int64_t* indices_dt = indices + o_plane * osizeH * osizeW;

  // For all output pixels...
  for (oh = ostartH; oh < oendH; oh += ostepH) {
    for (ow = ostartW; ow < oendW; ow += ostepW) {
      // Compute the gradients for the argmax input pixel
      const scalar_t* ptr_gradOutput = gradOutput_dt + oh * osizeW + ow;
      const int64_t* ptr_ind = indices_dt + oh * osizeW + ow;
      scalar_t grad_delta = *ptr_gradOutput;
      int64_t argmax = (*ptr_ind);
      gradInput_dt[argmax] += grad_delta;
    }
  }
}

template <typename scalar_t>
void adaptive_max_pool3d_backward_atomic_template(
    scalar_t* gradInput_data,
    const scalar_t* gradOutput_data,
    const int64_t* indices_data,
    int64_t totalZ,
    int isizeT,
    int isizeH,
    int isizeW,
    int osizeT,
    int osizeH,
    int osizeW) {
  int64_t offsetZ = 0;
  int width_group_size = 32;
  int height_group_size = 8;
  int height_group_range = std::max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    int width_group_range = totalZ > 65535 ? 65535 : totalZ;
    auto& queue = getCurrentSYCLQueue();
    sycl_kernel_submit<adaptive_max_pool3d_backward_atomic_kernel<scalar_t>>(
        sycl::range<2>{
            size_t(height_group_range * height_group_size),
            size_t(width_group_range * width_group_size),
        },
        sycl::range<2>{size_t(height_group_size), size_t(width_group_size)},
        queue,
        0,
        gradInput_data,
        gradOutput_data,
        indices_data,
        isizeT,
        isizeH,
        isizeW,
        osizeT,
        osizeH,
        osizeW,
        offsetZ);
    totalZ -= 65535;
    offsetZ += 65535;
  }
}

template <typename scalar_t>
void adaptive_max_pool3d_backward_template(
    scalar_t* gradInput_data,
    const scalar_t* gradOutput_data,
    const int64_t* indices_data,
    int64_t totalZ,
    int isizeT,
    int isizeH,
    int isizeW,
    int osizeT,
    int osizeH,
    int osizeW) {
  int64_t offsetZ = 0;
  int width_group_size = 32;
  int height_group_size = 8;
  int height_group_range = std::max((int)(16L / totalZ), 1);
  while (totalZ > 0) {
    int width_group_range = totalZ > 65535 ? 65535 : totalZ;
    auto& queue = getCurrentSYCLQueue();
    sycl_kernel_submit<adaptive_max_pool3d_backward_kernel_impl<scalar_t>>(
        sycl::range<2>{
            size_t(height_group_range * height_group_size),
            size_t(width_group_range * width_group_size),
        },
        sycl::range<2>{size_t(height_group_size), size_t(width_group_size)},
        queue,
        0,
        gradInput_data,
        gradOutput_data,
        indices_data,
        isizeT,
        isizeH,
        isizeW,
        osizeT,
        osizeH,
        osizeW,
        offsetZ);
    totalZ -= 65535;
    offsetZ += 65535;
  }
}

void adaptive_max_pool3d_backward_kernel(
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    const Tensor& gradInput) {
  const Tensor gradOutput_ = gradOutput.contiguous();

  gradInput.zero_();

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t osizeT, osizeH, osizeW;
  int64_t totalZ;

  if (input.ndimension() == 4) {
    sizeD = input.size(0);
    isizeT = input.size(1);
    isizeH = input.size(2);
    isizeW = input.size(3);

    osizeT = gradOutput_.size(1);
    osizeH = gradOutput_.size(2);
    osizeW = gradOutput_.size(3);
  } else {
    sizeD = input.size(1);
    isizeT = input.size(2);
    isizeH = input.size(3);
    isizeW = input.size(4);

    osizeT = gradOutput_.size(2);
    osizeH = gradOutput_.size(3);
    osizeW = gradOutput_.size(4);
  }

  bool atomic = (isizeW % osizeW != 0) || (isizeH % osizeH != 0) ||
      (isizeT % osizeT != 0);

  if (input.ndimension() == 4) {
    totalZ = sizeD * osizeT;
  } else {
    int sizeB = input.size(0);
    totalZ = sizeB * sizeD * osizeT;
  }

  if (atomic) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "adaptive_max_pool3d_backward_xpu",
        [&] {
          scalar_t* gradInput_data = gradInput.mutable_data_ptr<scalar_t>();
          const scalar_t* gradOutput_data =
              gradOutput_.const_data_ptr<scalar_t>();
          const int64_t* indices_data = indices.const_data_ptr<int64_t>();

          adaptive_max_pool3d_backward_atomic_template(
              gradInput_data,
              gradOutput_data,
              indices_data,
              totalZ,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "adaptive_max_pool3d_backward_xpu",
        [&] {
          scalar_t* gradInput_data = gradInput.mutable_data_ptr<scalar_t>();
          const scalar_t* gradOutput_data =
              gradOutput_.const_data_ptr<scalar_t>();
          const int64_t* indices_data = indices.const_data_ptr<int64_t>();

          adaptive_max_pool3d_backward_template(
              gradInput_data,
              gradOutput_data,
              indices_data,
              totalZ,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  }
}

} // namespace at::native::xpu

// clang-format off
DISABLE_RETURN_TYPE_WARNING_END
// clang-format on
