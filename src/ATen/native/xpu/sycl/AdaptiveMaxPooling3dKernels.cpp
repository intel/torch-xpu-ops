/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#ifdef _MSC_VER
  #pragma warning(push)
  #pragma warning(disable : 4715)
#endif

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
struct AdaptiveMaxPool3dKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    // iterators on output pixels
    int ot, oh, ow;

    int ostartH =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    int oendH = osizeH_;
    int ostepH = item.get_group_range(0) * item.get_local_range(0);
    int ostartW = item.get_local_id(1);
    int oendW = osizeW_;
    int ostepW = item.get_local_range(1);

    // select output plane
    int64_t o_plane = item.get_group(1) + offsetZ_;
    ot = o_plane % osizeT_;
    int d = o_plane / osizeT_;

    int istartT = start_index(ot, osizeT_, isizeT_);
    int iendT = end_index(ot, osizeT_, isizeT_);
    int kT = iendT - istartT;

    const scalar_t* input_dt =
        input_data_ + d * istrideD_ + istartT * istrideT_;

    scalar_t* output_dt = output_data_ + o_plane * osizeH_ * osizeW_;

    int64_t* indices_dt = indices_data_ + o_plane * osizeH_ * osizeW_;

    // For all output pixels...
    for (oh = ostartH; oh < oendH; oh += ostepH) {
      int istartH = start_index(oh, osizeH_, isizeH_);
      int iendH = end_index(oh, osizeH_, isizeH_);
      int kH = iendH - istartH;

      for (ow = ostartW; ow < oendW; ow += ostepW) {
        int istartW = start_index(ow, osizeW_, isizeW_);
        int iendW = end_index(ow, osizeW_, isizeW_);
        int kW = iendW - istartW;

        // Compute the average pooling from corresponding input pixels
        const scalar_t* ptr_input =
            input_dt + istartH * istrideH_ + istartW * istrideW_;
        scalar_t* ptr_output = output_dt + oh * osizeW_ + ow;
        int64_t* ptr_ind = indices_dt + oh * osizeW_ + ow;
        int64_t argmax =
            istartT * isizeH_ * isizeW_ + istartH * isizeW_ + istartW;
        scalar_t max = at::numeric_limits<scalar_t>::lower_bound(); // -Infinity

        int it, ih, iw;
        for (it = 0; it < kT; ++it) {
          for (ih = 0; ih < kH; ++ih) {
            for (iw = 0; iw < kW; ++iw) {
              scalar_t val = ptr_input[ih * istrideH_ + iw * istrideW_];
              if ((val > max) || at::_isnan(val)) {
                max = val;
                argmax = (it + istartT) * isizeH_ * isizeW_ +
                    (ih + istartH) * isizeW_ + iw + istartW;
              }
            }
          }
          ptr_input += istrideT_; // next input frame
        }
        // Update output and argmax
        *ptr_output = max;
        *ptr_ind = argmax;
      }
    }
  }

  AdaptiveMaxPool3dKernelFunctor(
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
      int64_t offsetZ)
      : input_data_(input_data),
        output_data_(output_data),
        indices_data_(indices_data),
        isizeT_(isizeT),
        isizeH_(isizeH),
        isizeW_(isizeW),
        osizeT_(osizeT),
        osizeH_(osizeH),
        osizeW_(osizeW),
        istrideD_(istrideD),
        istrideT_(istrideT),
        istrideH_(istrideH),
        istrideW_(istrideW),
        offsetZ_(offsetZ) {}

 private:
  const scalar_t* input_data_;
  scalar_t* output_data_;
  int64_t* indices_data_;
  int isizeT_;
  int isizeH_;
  int isizeW_;
  int osizeT_;
  int osizeH_;
  int osizeW_;
  int64_t istrideD_;
  int64_t istrideT_;
  int64_t istrideH_;
  int64_t istrideW_;
  int64_t offsetZ_;
};

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
    AdaptiveMaxPool3dKernelFunctor<scalar_t> kfn(
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
    auto& queue = getCurrentSYCLQueue();
    sycl_kernel_submit(
        sycl::range<2>{
            size_t(height_group_range * height_group_size),
            size_t(width_group_range * width_group_size),
        },
        sycl::range<2>{size_t(height_group_size), size_t(width_group_size)},
        queue,
        kfn);
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
struct AdaptiveMaxPool3dBackwardAtomicKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    // iterators on output pixels
    int oh, ow;

    int ostartH =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    int oendH = osizeH_;
    int ostepH = item.get_group_range(0) * item.get_local_range(0);
    int ostartW = item.get_local_id(1);
    int oendW = osizeW_;
    int ostepW = item.get_local_range(1);

    // select output plane
    int64_t o_plane = item.get_group(1) + offsetZ_;
    int d = o_plane / osizeT_;

    scalar_t* gradInput_dt = gradInput_ + d * isizeT_ * isizeH_ * isizeW_;
    const scalar_t* gradOutput_dt = gradOutput_ + o_plane * osizeH_ * osizeW_;
    const int64_t* indices_dt = indices_ + o_plane * osizeH_ * osizeW_;

    // For all output pixels...
    for (oh = ostartH; oh < oendH; oh += ostepH) {
      for (ow = ostartW; ow < oendW; ow += ostepW) {
        // Compute the gradients for the argmax input pixel
        const scalar_t* ptr_gradOutput = gradOutput_dt + oh * osizeW_ + ow;
        const int64_t* ptr_ind = indices_dt + oh * osizeW_ + ow;
        scalar_t grad_delta = *ptr_gradOutput;
        int64_t argmax = (*ptr_ind);
        atomicAdd(
            (sycl_global_ptr<scalar_t>)&(gradInput_dt[argmax]), grad_delta);
      }
    }
  }

  AdaptiveMaxPool3dBackwardAtomicKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* gradOutput,
      const int64_t* indices,
      int isizeT,
      int isizeH,
      int isizeW,
      int osizeT,
      int osizeH,
      int osizeW,
      int64_t offsetZ)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        indices_(indices),
        isizeT_(isizeT),
        isizeH_(isizeH),
        isizeW_(isizeW),
        osizeT_(osizeT),
        osizeH_(osizeH),
        osizeW_(osizeW),
        offsetZ_(offsetZ) {}

 private:
  scalar_t* gradInput_;
  const scalar_t* gradOutput_;
  const int64_t* indices_;
  int isizeT_;
  int isizeH_;
  int isizeW_;
  int osizeT_;
  int osizeH_;
  int osizeW_;
  int64_t istrideD_;
  int64_t istrideT_;
  int64_t istrideH_;
  int64_t istrideW_;
  int64_t offsetZ_;
};

template <typename scalar_t>
struct AdaptiveMaxPool3dBackwardKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    // iterators on output pixels
    int oh, ow;

    int ostartH =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    int oendH = osizeH_;
    int ostepH = item.get_group_range(0) * item.get_local_range(0);
    int ostartW = item.get_local_id(1);
    int oendW = osizeW_;
    int ostepW = item.get_local_range(1);

    // select output plane
    int64_t o_plane = item.get_group(1) + offsetZ_;
    int d = o_plane / osizeT_;

    scalar_t* gradInput_dt = gradInput_ + d * isizeT_ * isizeH_ * isizeW_;
    const scalar_t* gradOutput_dt = gradOutput_ + o_plane * osizeH_ * osizeW_;
    const int64_t* indices_dt = indices_ + o_plane * osizeH_ * osizeW_;

    // For all output pixels...
    for (oh = ostartH; oh < oendH; oh += ostepH) {
      for (ow = ostartW; ow < oendW; ow += ostepW) {
        // Compute the gradients for the argmax input pixel
        const scalar_t* ptr_gradOutput = gradOutput_dt + oh * osizeW_ + ow;
        const int64_t* ptr_ind = indices_dt + oh * osizeW_ + ow;
        scalar_t grad_delta = *ptr_gradOutput;
        int64_t argmax = (*ptr_ind);
        gradInput_dt[argmax] += grad_delta;
      }
    }
  }

  AdaptiveMaxPool3dBackwardKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* gradOutput,
      const int64_t* indices,
      int isizeT,
      int isizeH,
      int isizeW,
      int osizeT,
      int osizeH,
      int osizeW,
      int64_t offsetZ)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        indices_(indices),
        isizeT_(isizeT),
        isizeH_(isizeH),
        isizeW_(isizeW),
        osizeT_(osizeT),
        osizeH_(osizeH),
        osizeW_(osizeW),
        offsetZ_(offsetZ) {}

 private:
  scalar_t* gradInput_;
  const scalar_t* gradOutput_;
  const int64_t* indices_;
  int isizeT_;
  int isizeH_;
  int isizeW_;
  int osizeT_;
  int osizeH_;
  int osizeW_;
  int64_t istrideD_;
  int64_t istrideT_;
  int64_t istrideH_;
  int64_t istrideW_;
  int64_t offsetZ_;
};

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
    AdaptiveMaxPool3dBackwardAtomicKernelFunctor<scalar_t> kfn(
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
    auto& queue = getCurrentSYCLQueue();
    sycl_kernel_submit(
        sycl::range<2>{
            size_t(height_group_range * height_group_size),
            size_t(width_group_range * width_group_size),
        },
        sycl::range<2>{size_t(height_group_size), size_t(width_group_size)},
        queue,
        kfn);
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
    AdaptiveMaxPool3dBackwardKernelFunctor<scalar_t> kfn(
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
    auto& queue = getCurrentSYCLQueue();
    sycl_kernel_submit(
        sycl::range<2>{
            size_t(height_group_range * height_group_size),
            size_t(width_group_range * width_group_size),
        },
        sycl::range<2>{size_t(height_group_size), size_t(width_group_size)},
        queue,
        kfn);
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

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
#ifdef _MSC_VER
  #pragma warning(pop)
#endif

