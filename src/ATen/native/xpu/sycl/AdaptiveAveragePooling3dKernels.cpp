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
#include <ATen/NumericUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/AdaptivePooling.h>
#include <ATen/native/xpu/sycl/AdaptiveAveragePooling3dKernels.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

inline int64_t start_index(int64_t a, int64_t b, int64_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}

inline int64_t end_index(int64_t a, int64_t b, int64_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}

template <typename scalar_t, typename accscalar_t>
struct AdaptiveAvgPool3dKernelFunctor {
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
        accscalar_t sum = static_cast<accscalar_t>(0);

        int it, ih, iw;
        for (it = 0; it < kT; ++it) {
          for (ih = 0; ih < kH; ++ih) {
            for (iw = 0; iw < kW; ++iw) {
              scalar_t val = ptr_input[ih * istrideH_ + iw * istrideW_];
              sum += static_cast<accscalar_t>(val);
            }
          }
          ptr_input += istrideT_; // next input frame
        }
        // Update output
        const accscalar_t divide_factor =
            static_cast<accscalar_t>(kT * kH * kW);
        *ptr_output = static_cast<scalar_t>(sum / divide_factor);
      }
    }
  }

  AdaptiveAvgPool3dKernelFunctor(
      const scalar_t* input_data,
      scalar_t* output_data,
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

template <typename scalar_t, typename accscalar_t>
void adaptive_avg_pool3d_template(
    const scalar_t* input_data,
    scalar_t* output_data,
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
    AdaptiveAvgPool3dKernelFunctor<scalar_t, accscalar_t> kfn(
        input_data,
        output_data,
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
void adaptive_avg_pool3d_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef& output_size) {
  TensorArg output_arg{output, "output", 1};
  TensorArg input_arg{input_, "input_", 2};
  checkAllSameGPU("adaptive_avg_pool3d_xpu", {output_arg, input_arg});

  for (int64_t i = 1; i < input_.ndimension(); i++) {
    TORCH_CHECK(
        input_.size(i) > 0,
        "adaptive_avg_pool3d_xpu(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input_.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      (input_.ndimension() == 4 || input_.ndimension() == 5),
      "adaptive_avg_pool3d_xpu(): Expected 4D or 5D tensor, but got ",
      input_.sizes());

  // the jit sometimes passes output_size.size() == 1
  TORCH_CHECK(
      output_size.size() == 1 || output_size.size() == 3,
      "adaptive_avg_pool3d: internal error: output_size.size() must be 1 or 3");

  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t istrideD, istrideT, istrideH, istrideW;
  int64_t totalZ;

  const Tensor& input = input_.ndimension() == 4 ? input_ : input_.contiguous();

  if (input.ndimension() == 4) {
    sizeD = input.size(0);
    isizeT = input.size(1);
    isizeH = input.size(2);
    isizeW = input.size(3);

    istrideD = input.stride(0);
    istrideT = input.stride(1);
    istrideH = input.stride(2);
    istrideW = input.stride(3);

    output.resize_({sizeD, osizeT, osizeH, osizeW});

    totalZ = sizeD * osizeT;
  } else {
    int64_t sizeB = input.size(0);
    sizeD = input.size(1);
    isizeT = input.size(2);
    isizeH = input.size(3);
    isizeW = input.size(4);

    istrideD = input.stride(1);
    istrideT = input.stride(2);
    istrideH = input.stride(3);
    istrideW = input.stride(4);

    output.resize_({sizeB, sizeD, osizeT, osizeH, osizeW});

    totalZ = sizeB * sizeD * osizeT;
  }

  if (output.numel() == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "adaptive_avg_pool3d_xpu", [&] {
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
        const scalar_t* input_data = input.const_data_ptr<scalar_t>();
        scalar_t* output_data = output.mutable_data_ptr<scalar_t>();

        adaptive_avg_pool3d_template<scalar_t, accscalar_t>(
            input_data,
            output_data,
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
struct AdaptiveAvgPool3dBackwardAtomicKernelFunctor {
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
    ot = o_plane % osizeT_; // output frame/time
    int d = o_plane / osizeT_;

    int istartT = start_index(ot, osizeT_, isizeT_);
    int iendT = end_index(ot, osizeT_, isizeT_);
    int kT = iendT - istartT;

    // gradInput offset by slice/feature and earliest relevant frame/time
    scalar_t* gradInput_nt =
        gradInput_ + (d * isizeT_ + istartT) * isizeH_ * isizeW_;
    // gradOutput offset by slice/feature and frame/time
    const scalar_t* gradOutput_nt = gradOutput_ + o_plane * osizeH_ * osizeW_;

    // For all output pixels...
    for (oh = ostartH; oh < oendH; oh += ostepH) {
      int istartH = start_index(oh, osizeH_, isizeH_);
      int iendH = end_index(oh, osizeH_, isizeH_);
      int kH = iendH - istartH;

      for (ow = ostartW; ow < oendW; ow += ostepW) {
        int istartW = start_index(ow, osizeW_, isizeW_);
        int iendW = end_index(ow, osizeW_, isizeW_);
        int kW = iendW - istartW;

        // Compute the gradients from corresponding input pixels
        scalar_t* ptr_gradInput = gradInput_nt + istartH * isizeW_ + istartW;
        const scalar_t* ptr_gradOutput = gradOutput_nt + oh * osizeW_ + ow;
        scalar_t grad_delta = *ptr_gradOutput / kT / kH / kW;

        int it, ih, iw;
        for (it = 0; it < kT; ++it) {
          for (ih = 0; ih < kH; ++ih) {
            for (iw = 0; iw < kW; ++iw) {
              atomicAdd(
                  (sycl_global_ptr<scalar_t>)&(
                      ptr_gradInput[ih * isizeW_ + iw]),
                  grad_delta);
            }
          }
          ptr_gradInput += isizeH_ * isizeW_; // next input frame
        }
      }
    }
  }

  AdaptiveAvgPool3dBackwardAtomicKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* gradOutput,
      int isizeT,
      int isizeH,
      int isizeW,
      int osizeT,
      int osizeH,
      int osizeW,
      int64_t offsetZ)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
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

template <typename scalar_t, typename accscalar_t>
struct AdaptiveAvgPool3dBackwardKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    // iterators on output pixels
    int it, ih, iw;

    int istartH =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    int iendH = isizeH_;
    int istepH = item.get_group_range(0) * item.get_local_range(0);
    int istartW = item.get_local_id(1);
    int iendW = isizeW_;
    int istepW = item.get_local_range(1);

    // select output plane
    int64_t i_plane = item.get_group(1) + offsetZ_;
    it = i_plane % isizeT_; // output frame/time
    int d = i_plane / isizeT_;

    int ostartT = start_index(it, isizeT_, osizeT_);
    int oendT = end_index(it, isizeT_, osizeT_);

    // gradInput offset by slice/feature and frame/time.
    scalar_t* gradInput_dt = gradInput_ + i_plane * isizeH_ * isizeW_;
    // gradOutput offset by slice/feature and earliest relevant frame/time
    const scalar_t* gradOutput_dt =
        gradOutput_ + (d * osizeT_ + ostartT) * osizeH_ * osizeW_;
    // For all input pixels...
    for (ih = istartH; ih < iendH; ih += istepH) {
      int ostartH = start_index(ih, isizeH_, osizeH_);
      int oendH = end_index(ih, isizeH_, osizeH_);

      for (iw = istartW; iw < iendW; iw += istepW) {
        int ostartW = start_index(iw, isizeW_, osizeW_);
        int oendW = end_index(iw, isizeW_, osizeW_);

        // Compute the gradients from corresponding output pixels
        scalar_t* ptr_gradInput = gradInput_dt + ih * isizeW_ + iw;
        const scalar_t* ptr_gradOutput = gradOutput_dt;

        // for all relevant output pixels
        int ot, oh, ow;
        for (ot = ostartT; ot < oendT; ++ot) {
          int kT = end_index(ot, osizeT_, isizeT_) -
              start_index(ot, osizeT_, isizeT_);
          for (oh = ostartH; oh < oendH; ++oh) {
            int kH = end_index(oh, osizeH_, isizeH_) -
                start_index(oh, osizeH_, isizeH_);
            for (ow = ostartW; ow < oendW; ++ow) {
              int kW = end_index(ow, osizeW_, isizeW_) -
                  start_index(ow, osizeW_, isizeW_);
              const accscalar_t divide_factor = kW * kH * kT;
              accscalar_t grad_delta = static_cast<accscalar_t>(
                  ptr_gradOutput[oh * osizeW_ + ow] / divide_factor);
              *ptr_gradInput += static_cast<scalar_t>(grad_delta);
            }
          }
          ptr_gradOutput += osizeH_ * osizeW_; // next output frame
        }
      }
    }
  }

  AdaptiveAvgPool3dBackwardKernelFunctor(
      scalar_t* gradInput,
      const scalar_t* gradOutput,
      int isizeT,
      int isizeH,
      int isizeW,
      int osizeT,
      int osizeH,
      int osizeW,
      int64_t offsetZ)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
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
void adaptive_avg_pool3d_backward_atomic_template(
    scalar_t* gradInput_data,
    const scalar_t* gradOutput_data,
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
    AdaptiveAvgPool3dBackwardAtomicKernelFunctor<scalar_t> kfn(
        gradInput_data,
        gradOutput_data,
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

template <typename scalar_t, typename accscalar_t>
void adaptive_avg_pool3d_backward_template(
    scalar_t* gradInput_data,
    const scalar_t* gradOutput_data,
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
    AdaptiveAvgPool3dBackwardKernelFunctor<scalar_t, accscalar_t> kfn(
        gradInput_data,
        gradOutput_data,
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

void adaptive_avg_pool3d_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  TensorArg grad_input_arg{gradInput, "gradInput", 1};
  TensorArg grad_output_arg{gradOutput_, "gradOutput_", 2};
  TensorArg input_arg{input, "input", 3};

  adaptive_pool_empty_output_check(gradOutput_, "adaptive_avg_pool3d_backward");
  TORCH_CHECK(
      input.dim() == gradOutput_.dim(),
      __func__,
      ": Expected dimensions ",
      input.dim(),
      " for `gradOutput_` but got dimensions ",
      gradOutput_.dim());

  checkAllSameGPU(
      "adaptive_avg_pool3d_backward_xpu",
      {grad_input_arg, grad_output_arg, input_arg});

  const Tensor gradOutput = gradOutput_.contiguous();

  gradInput.resize_as_(input);
  if (gradInput.numel() == 0) {
    return;
  }

  gradInput.zero_();

  int64_t sizeD, isizeT, isizeH, isizeW;
  int64_t osizeT, osizeH, osizeW;
  int64_t totalZ;

  if (input.ndimension() == 4) {
    sizeD = input.size(0);
    isizeT = input.size(1);
    isizeH = input.size(2);
    isizeW = input.size(3);

    osizeT = gradOutput.size(1);
    osizeH = gradOutput.size(2);
    osizeW = gradOutput.size(3);
  } else {
    sizeD = input.size(1);
    isizeT = input.size(2);
    isizeH = input.size(3);
    isizeW = input.size(4);

    osizeT = gradOutput.size(2);
    osizeH = gradOutput.size(3);
    osizeW = gradOutput.size(4);
  }

  bool atomic = (isizeW % osizeW != 0) || (isizeH % osizeH != 0) ||
      (isizeT % osizeT != 0);

  if (input.ndimension() == 4) {
    totalZ = atomic ? sizeD * osizeT : sizeD * isizeT;
  } else {
    int sizeB = input.size(0);
    totalZ = atomic ? sizeB * sizeD * osizeT : sizeB * sizeD * isizeT;
  }

  if (atomic) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        input.scalar_type(),
        "adaptive_avg_pool3d_backward_xpu",
        [&] {
          scalar_t* gradInput_data = gradInput.mutable_data_ptr<scalar_t>();
          const scalar_t* gradOutput_data =
              gradOutput.const_data_ptr<scalar_t>();

          adaptive_avg_pool3d_backward_atomic_template(
              gradInput_data,
              gradOutput_data,
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
          using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
          scalar_t* gradInput_data = gradInput.mutable_data_ptr<scalar_t>();
          const scalar_t* gradOutput_data =
              gradOutput.const_data_ptr<scalar_t>();

          adaptive_avg_pool3d_backward_template<scalar_t, accscalar_t>(
              gradInput_data,
              gradOutput_data,
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

