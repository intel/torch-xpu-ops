/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Pool.h>
#include <ATen/native/xpu/sycl/Atomics.h>

#include <ATen/native/xpu/sycl/AveragePool3dKernels.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

inline int min(int a, int b) {
  return a <= b ? a : b;
}

inline int max(int a, int b) {
  return a >= b ? a : b;
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool3dKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    index_t oCol = item.get_global_id()[2];
    index_t oRow = item.get_global_id()[1];
    index_t oFrame = (item.get_group(0) + offsetZ_) % output_.size(1);
    index_t slice = (item.get_group(0) + offsetZ_) / output_.size(1);
    auto out_data = output_;
    if (oRow < out_data.size(2) && oCol < out_data.size(3)) {
      accscalar_t sum = 0.0;

      index_t tstart = oFrame * dT_ - padT_;
      index_t hstart = oRow * dH_ - padH_;
      index_t wstart = oCol * dW_ - padW_;
      index_t tend = min(tstart + kT_, input_.size(1) + padT_);
      index_t hend = min(hstart + kH_, input_.size(2) + padH_);
      index_t wend = min(wstart + kW_, input_.size(3) + padW_);
      index_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);

      tstart = max(tstart, 0);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
      tend = min(tend, input_.size(1));
      hend = min(hend, input_.size(2));
      wend = min(wend, input_.size(3));

      if (tstart >= tend || hstart >= hend || wstart >= wend) {
        out_data[slice][oFrame][oRow][oCol] = static_cast<scalar_t>(0.0);
        return;
      }

      accscalar_t divide_factor;
      if (divisor_override_) {
        divide_factor = static_cast<accscalar_t>(divisor_override_);
      } else {
        if (count_include_pad_) {
          divide_factor = static_cast<accscalar_t>(pool_size);
        } else {
          divide_factor = static_cast<accscalar_t>(
              (tend - tstart) * (hend - hstart) * (wend - wstart));
        }
      }

      index_t ti, hi, wi;
      for (ti = tstart; ti < tend; ++ti) {
        for (hi = hstart; hi < hend; ++hi) {
          for (wi = wstart; wi < wend; ++wi) {
            const scalar_t val = input_[slice][ti][hi][wi];
            sum += val;
          }
        }
      }

      out_data[slice][oFrame][oRow][oCol] =
          static_cast<scalar_t>(sum / divide_factor);
    }
  }
  AvgPool3dKernelFunctor(
      int kT,
      int kH,
      int kW,
      int dT,
      int dH,
      int dW,
      int padT,
      int padH,
      int padW,
      bool count_include_pad,
      int offsetZ,
      int divisor_override,
      PackedTensorAccessor64<const scalar_t, 4> input_acc,
      PackedTensorAccessor64<scalar_t, 4> output_acc)
      : kT_(kT),
        kH_(kH),
        kW_(kW),
        dT_(dT),
        dH_(dH),
        dW_(dW),
        padT_(padT),
        padH_(padH),
        padW_(padW),
        count_include_pad_(count_include_pad),
        offsetZ_(offsetZ),
        divisor_override_(divisor_override),
        input_(input_acc),
        output_(output_acc) {}

 private:
  int kT_;
  int kH_;
  int kW_;
  int dT_;
  int dH_;
  int dW_;
  int padT_;
  int padH_;
  int padW_;
  bool count_include_pad_;
  int offsetZ_;
  int divisor_override_;
  PackedTensorAccessor64<const scalar_t, 4> input_;
  PackedTensorAccessor64<scalar_t, 4> output_;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void avg_pool3d_out_template(
    Tensor& work_input,
    Tensor& work_output,
    const int kT,
    const int kH,
    const int kW,
    const int dT,
    const int dH,
    const int dW,
    const int padT,
    const int padH,
    const int padW,
    const bool count_include_pad,
    const int offsetZ,
    const int totalZ,
    const int divisor_override) {
  auto input_acc = work_input.packed_accessor64<const scalar_t, 4>();
  auto output_acc = work_output.packed_accessor64<scalar_t, 4>();
  AvgPool3dKernelFunctor<scalar_t, accscalar_t, index_t> kfn(
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      padT,
      padH,
      padW,
      count_include_pad,
      offsetZ,
      divisor_override,
      input_acc,
      output_acc);

  // width size is fixed size = 32, height dim equals =
  // syclMaxWorkGroupSize(kfn) / width_size
  index_t width_group_size = 32;
  index_t height_group_size = syclMaxWorkGroupSize(kfn) / width_group_size;
  index_t width_group_range =
      ceil_div<index_t>(work_output.size(-1), width_group_size);
  index_t height_group_range =
      ceil_div<index_t>(work_output.size(-2), height_group_size);

  index_t z_group_range = totalZ > 65535 ? 65535 : totalZ;
  auto& queue = getCurrentSYCLQueue();
  sycl_kernel_submit(
      sycl::range<3>{
          size_t(z_group_range),
          size_t(height_group_range * height_group_size),
          size_t(width_group_range * width_group_size),
      },
      sycl::range<3>{1, size_t(height_group_size), size_t(width_group_size)},
      queue,
      kfn);
}

void avg_pool3d_kernel(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    const Tensor& output) {
  TensorArg output_arg{output, "output", 1};
  TensorArg input_arg{input, "input", 2};

  checkAllSameGPU(__func__, {output_arg, input_arg});

  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  // if divisor==0 then we will ignore it
  int64_t divisor = 0;
  if (divisor_override.has_value()) {
    divisor = divisor_override.value();
  }

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t otime =
      pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight =
      pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth =
      pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  Tensor work_input = input.contiguous();
  Tensor work_output = output;

  if (input.ndimension() == 5) {
    // Collapse batch and feature dimensions.
    work_input = work_input.reshape({nbatch * nslices, itime, iheight, iwidth});
    work_output =
        work_output.reshape({nbatch * nslices, otime, oheight, owidth});
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "avg_pool3d_xpu", [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        int64_t totalZ = otime * nslices * nbatch;
        int64_t offsetZ = 0;
        while (totalZ > 0) {
          if (canUse32BitIndexMath(input)) {
            avg_pool3d_out_template<scalar_t, accscalar_t, int32_t>(
                work_input,
                work_output,
                kT,
                kH,
                kW,
                dT,
                dH,
                dW,
                padT,
                padH,
                padW,
                count_include_pad,
                offsetZ,
                totalZ,
                divisor);
          } else {
            avg_pool3d_out_template<scalar_t, accscalar_t, int64_t>(
                work_input,
                work_output,
                kT,
                kH,
                kW,
                dT,
                dH,
                dW,
                padT,
                padH,
                padW,
                count_include_pad,
                offsetZ,
                totalZ,
                divisor);
          }
          totalZ -= 65535;
          offsetZ += 65535;
        }
      });
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool3dBackwardStride1KernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    index_t iCol = item.get_global_id()[2];
    index_t iRow = item.get_global_id()[1];
    index_t iFrame = (item.get_group(0) + offsetZ_) % grad_input_.size(1);
    index_t slice = (item.get_group(0) + offsetZ_) / grad_input_.size(1);

    auto grad_input_data = grad_input_;
    if (iRow < grad_input_.size(2) && iCol < grad_input_.size(3)) {
      accscalar_t sum = 0.0;
      const scalar_t* gOut = &grad_output_[slice][max(0, iFrame - kT_ + 1)][max(
          0, iRow - kH_ + 1)][max(0, iCol - kW_ + 1)];
      index_t frameOffset = 0;
      for (index_t oFrame = max(0, iFrame - kT_ + 1);
           oFrame < min(iFrame + 1, grad_output_.size(1));
           ++oFrame) {
        index_t rowOffset = frameOffset;
        for (index_t oRow = max(0, iRow - kH_ + 1);
             oRow < min(iRow + 1, grad_output_.size(2));
             ++oRow) {
          index_t colOffset = rowOffset;
          for (index_t oCol = max(0, iCol - kW_ + 1);
               oCol < min(iCol + 1, grad_output_.size(3));
               ++oCol) {
            sum += gOut[colOffset];
            ++colOffset;
          }
          rowOffset += grad_output_.size(3);
        }
        frameOffset += grad_output_.size(2) * grad_output_.size(3);
      }
      grad_input_data[slice][iFrame][iRow][iCol] =
          static_cast<scalar_t>(sum * normFactor_);
    }
  }
  AvgPool3dBackwardStride1KernelFunctor(
      int kT,
      int kH,
      int kW,
      accscalar_t normFactor,
      int offsetZ,
      PackedTensorAccessor64<const scalar_t, 4> grad_output,
      PackedTensorAccessor64<scalar_t, 4> grad_input)
      : kT_(kT),
        kH_(kH),
        kW_(kW),
        normFactor_(normFactor),
        offsetZ_(offsetZ),
        grad_output_(grad_output),
        grad_input_(grad_input) {}

 private:
  int kT_;
  int kH_;
  int kW_;
  accscalar_t normFactor_;
  int offsetZ_;
  PackedTensorAccessor64<const scalar_t, 4> grad_output_;
  PackedTensorAccessor64<scalar_t, 4> grad_input_;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void avg_pool3d_backward_stride1_template(
    const Tensor& grad_output,
    Tensor& grad_input,
    int kT,
    int kH,
    int kW,
    accscalar_t normFactor,
    int offsetZ,
    int totalZ) {
  auto grad_output_acc = grad_output.packed_accessor64<const scalar_t, 4>();
  auto grad_input_acc = grad_input.packed_accessor64<scalar_t, 4>();
  AvgPool3dBackwardStride1KernelFunctor<scalar_t, accscalar_t, index_t> kfn(
      kT, kH, kW, normFactor, offsetZ, grad_output_acc, grad_input_acc);
  // width size is fixed size = 32, height dim equals =
  // syclMaxWorkGroupSize(kfn) / width_size
  index_t width_group_size = 32;
  index_t height_group_size = syclMaxWorkGroupSize(kfn) / width_group_size;
  index_t width_group_range =
      ceil_div<index_t>(grad_input.size(-1), width_group_size);
  index_t height_group_range =
      ceil_div<index_t>(grad_input.size(-2), height_group_size);

  index_t z_group_range = totalZ > 65535 ? 65535 : totalZ;

  auto& queue = getCurrentSYCLQueue();
  sycl_kernel_submit(
      sycl::range<3>{
          size_t(z_group_range),
          size_t(height_group_range * height_group_size),
          size_t(width_group_range * width_group_size),
      },
      sycl::range<3>{1, size_t(height_group_size), size_t(width_group_size)},
      queue,
      kfn);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool3dBackwardAtomicKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    index_t oCol = item.get_global_id()[2];
    index_t oRow = item.get_global_id()[1];
    index_t oFrame = (item.get_group(0) + offsetZ_) % grad_output_.size(1);
    index_t slice = (item.get_group(0) + offsetZ_) / grad_output_.size(1);

    auto grad_input_data = grad_input_;
    if (oRow < grad_output_.size(2) && oCol < grad_output_.size(3)) {
      index_t tstart = oFrame * dT_ - padT_;
      index_t hstart = oRow * dH_ - padH_;
      index_t wstart = oCol * dW_ - padW_;
      index_t tend = min(tstart + kT_, grad_input_.size(1) + padT_);
      index_t hend = min(hstart + kH_, grad_input_.size(2) + padH_);
      index_t wend = min(wstart + kW_, grad_input_.size(3) + padW_);
      index_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
      tstart = max(tstart, 0);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
      tend = min(tend, grad_input_.size(1));
      hend = min(hend, grad_input_.size(2));
      wend = min(wend, grad_input_.size(3));

      accscalar_t divide_factor;
      if (divisor_override_) {
        divide_factor = static_cast<accscalar_t>(divisor_override_);
      } else {
        if (count_include_pad_) {
          divide_factor = static_cast<accscalar_t>(pool_size);
        } else {
          divide_factor = static_cast<accscalar_t>(
              (tend - tstart) * (hend - hstart) * (wend - wstart));
        }
      }

      scalar_t val = static_cast<scalar_t>(
          static_cast<accscalar_t>(grad_output_[slice][oFrame][oRow][oCol]) /
          divide_factor);

      for (index_t iFrame = tstart; iFrame < tend; ++iFrame) {
        for (index_t iRow = hstart; iRow < hend; ++iRow) {
          for (index_t iCol = wstart; iCol < wend; ++iCol) {
            const index_t index = slice * grad_input_.stride(0) +
                iFrame * grad_input_.stride(1) + iRow * grad_input_.stride(2) +
                iCol * grad_input_.stride(3);
            atomicAdd(
                (sycl_global_ptr<scalar_t>)&grad_input_data.data()[index], val);
          }
        }
      }
    }
  }
  AvgPool3dBackwardAtomicKernelFunctor(
      int kT,
      int kH,
      int kW,
      int dT,
      int dH,
      int dW,
      int padT,
      int padH,
      int padW,
      bool count_include_pad,
      int offsetZ,
      int divisor_override,
      PackedTensorAccessor64<const scalar_t, 4> grad_output,
      PackedTensorAccessor64<scalar_t, 4> grad_input)
      : kT_(kT),
        kH_(kH),
        kW_(kW),
        dT_(dT),
        dH_(dH),
        dW_(dW),
        padT_(padT),
        padH_(padH),
        padW_(padW),
        count_include_pad_(count_include_pad),
        offsetZ_(offsetZ),
        divisor_override_(divisor_override),
        grad_output_(grad_output),
        grad_input_(grad_input) {}

 private:
  int kT_;
  int kH_;
  int kW_;
  int dT_;
  int dH_;
  int dW_;
  int padT_;
  int padH_;
  int padW_;
  bool count_include_pad_;
  int offsetZ_;
  int divisor_override_;
  PackedTensorAccessor64<const scalar_t, 4> grad_output_;
  PackedTensorAccessor64<scalar_t, 4> grad_input_;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void avg_pool3d_backward_atomic_template(
    const Tensor& grad_output,
    Tensor& grad_input,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool count_include_pad,
    int offsetZ,
    int totalZ,
    int divisor_override) {
  auto grad_output_acc = grad_output.packed_accessor64<const scalar_t, 4>();
  auto grad_input_acc = grad_input.packed_accessor64<scalar_t, 4>();

  AvgPool3dBackwardAtomicKernelFunctor<scalar_t, accscalar_t, index_t> kfn(
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      padT,
      padH,
      padW,
      count_include_pad,
      offsetZ,
      divisor_override,
      grad_output_acc,
      grad_input_acc);

  // width size is fixed size = 32, height dim equals =
  // syclMaxWorkGroupSize(kfn) / width_size
  index_t width_group_size = 32;
  index_t height_group_size = syclMaxWorkGroupSize(kfn) / width_group_size;
  index_t width_group_range =
      ceil_div<index_t>(grad_output.size(-1), width_group_size);
  index_t height_group_range =
      ceil_div<index_t>(grad_output.size(-2), height_group_size);

  index_t z_group_range = totalZ > 65535 ? 65535 : totalZ;

  auto& queue = getCurrentSYCLQueue();
  sycl_kernel_submit(
      sycl::range<3>{
          size_t(z_group_range),
          size_t(height_group_range * height_group_size),
          size_t(width_group_range * width_group_size),
      },
      sycl::range<3>{1, size_t(height_group_size), size_t(width_group_size)},
      queue,
      kfn);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool3dBackwardKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    index_t oCol = item.get_global_id()[2];
    index_t oRow = item.get_global_id()[1];
    index_t oFrame = (item.get_group(0) + offsetZ_) % grad_output_.size(1);
    index_t slice = (item.get_group(0) + offsetZ_) / grad_output_.size(1);

    auto grad_input_data = grad_input_;
    if (oRow < grad_output_.size(2) && oCol < grad_output_.size(3)) {
      index_t tstart = oFrame * dT_ - padT_;
      index_t hstart = oRow * dH_ - padH_;
      index_t wstart = oCol * dW_ - padW_;
      index_t tend = min(tstart + kT_, grad_input_.size(1) + padT_);
      index_t hend = min(hstart + kH_, grad_input_.size(2) + padH_);
      index_t wend = min(wstart + kW_, grad_input_.size(3) + padW_);
      index_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
      tstart = max(tstart, 0);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
      tend = min(tend, grad_input_.size(1));
      hend = min(hend, grad_input_.size(2));
      wend = min(wend, grad_input_.size(3));

      accscalar_t divide_factor;
      if (divisor_override_) {
        divide_factor = static_cast<accscalar_t>(divisor_override_);
      } else {
        if (count_include_pad_) {
          divide_factor = static_cast<accscalar_t>(pool_size);
        } else {
          divide_factor = static_cast<accscalar_t>(
              (tend - tstart) * (hend - hstart) * (wend - wstart));
        }
      }

      scalar_t val = static_cast<scalar_t>(
          static_cast<accscalar_t>(grad_output_[slice][oFrame][oRow][oCol]) /
          divide_factor);

      for (index_t iFrame = tstart; iFrame < tend; ++iFrame) {
        for (index_t iRow = hstart; iRow < hend; ++iRow) {
          for (index_t iCol = wstart; iCol < wend; ++iCol) {
            grad_input_data[slice][iFrame][iRow][iCol] = val;
          }
        }
      }
    }
  }
  AvgPool3dBackwardKernelFunctor(
      int kT,
      int kH,
      int kW,
      int dT,
      int dH,
      int dW,
      int padT,
      int padH,
      int padW,
      bool count_include_pad,
      int offsetZ,
      int divisor_override,
      PackedTensorAccessor64<const scalar_t, 4> grad_output,
      PackedTensorAccessor64<scalar_t, 4> grad_input)
      : kT_(kT),
        kH_(kH),
        kW_(kW),
        dT_(dT),
        dH_(dH),
        dW_(dW),
        padT_(padT),
        padH_(padH),
        padW_(padW),
        count_include_pad_(count_include_pad),
        offsetZ_(offsetZ),
        divisor_override_(divisor_override),
        grad_output_(grad_output),
        grad_input_(grad_input) {}

 private:
  int kT_;
  int kH_;
  int kW_;
  int dT_;
  int dH_;
  int dW_;
  int padT_;
  int padH_;
  int padW_;
  bool count_include_pad_;
  int offsetZ_;
  int divisor_override_;
  PackedTensorAccessor64<const scalar_t, 4> grad_output_;
  PackedTensorAccessor64<scalar_t, 4> grad_input_;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void avg_pool3d_backward_template(
    const Tensor& grad_output,
    Tensor& grad_input,
    int kT,
    int kH,
    int kW,
    int dT,
    int dH,
    int dW,
    int padT,
    int padH,
    int padW,
    bool count_include_pad,
    int offsetZ,
    int totalZ,
    int divisor_override) {
  auto grad_output_acc = grad_output.packed_accessor64<const scalar_t, 4>();
  auto grad_input_acc = grad_input.packed_accessor64<scalar_t, 4>();

  AvgPool3dBackwardKernelFunctor<scalar_t, accscalar_t, index_t> kfn(
      kT,
      kH,
      kW,
      dT,
      dH,
      dW,
      padT,
      padH,
      padW,
      count_include_pad,
      offsetZ,
      divisor_override,
      grad_output_acc,
      grad_input_acc);

  // width size is fixed size = 32, height dim equals =
  // syclMaxWorkGroupSize(kfn) / width_size
  index_t width_group_size = 32;
  index_t height_group_size = syclMaxWorkGroupSize(kfn) / width_group_size;
  index_t width_group_range =
      ceil_div<index_t>(grad_output.size(-1), width_group_size);
  index_t height_group_range =
      ceil_div<index_t>(grad_output.size(-2), height_group_size);

  index_t z_group_range = totalZ > 65535 ? 65535 : totalZ;

  auto& queue = getCurrentSYCLQueue();
  sycl_kernel_submit(
      sycl::range<3>{
          size_t(z_group_range),
          size_t(height_group_range * height_group_size),
          size_t(width_group_range * width_group_size),
      },
      sycl::range<3>{1, size_t(height_group_size), size_t(width_group_size)},
      queue,
      kfn);
}

void avg_pool3d_backward_kernel(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    std::optional<int64_t> divisor_override,
    const Tensor& gradInput) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("avg_pool3d_backward_xpu");

  TensorArg gradInput_arg{gradInput, "gradInput", 1};
  TensorArg gradOutput_arg{gradOutput, "gradOutput", 2};
  TensorArg input_arg{input, "input", 3};

  checkAllSameGPU(__func__, {gradInput_arg, gradOutput_arg, input_arg});

  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1
      ? kT
      : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dT
                                : safe_downcast<int, int64_t>(stride[2]);

  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW =
      padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(
      (gradOutput.ndimension() == 4 || gradOutput.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for gradOutput");

  // if divisor==0 then we will ignore it
  int64_t divisor = 0;
  if (divisor_override.has_value()) {
    divisor = divisor_override.value();
  }

  gradInput.zero_();

  const int64_t nbatch = input.ndimension() == 5 ? input.size(-5) : 1;
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t otime = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

  const bool kernelsOverlap = (dT < kT) || (dH < kH) || (dW < kW);

  Tensor work_grad_input = gradInput;
  Tensor work_grad_output = gradOutput.contiguous();

  if (input.ndimension() == 5) {
    // Collapse batch and feature dimensions.
    work_grad_input =
        work_grad_input.reshape({nbatch * nslices, itime, iheight, iwidth});
    work_grad_output =
        work_grad_output.reshape({nbatch * nslices, otime, oheight, owidth});
  }
  if (canUse32BitIndexMath(work_grad_input)) {
    if (dT == 1 && dH == 1 && dW == 1 && padT == 0 && padH == 0 && padW == 0) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          input.scalar_type(),
          "avg_pool3d_backward_stride1_xpu",
          [&] {
            using accscalar_t = acc_type_device<scalar_t, kXPU>;
            int64_t totalZ = itime * nslices * nbatch;
            int64_t offsetZ = 0;

            accscalar_t divide_factor;
            if (divisor) {
              divide_factor = static_cast<accscalar_t>(divisor);
            } else {
              divide_factor = static_cast<accscalar_t>(kT * kH * kW);
            }

            while (totalZ > 0) {
              avg_pool3d_backward_stride1_template<
                  scalar_t,
                  accscalar_t,
                  int32_t>(
                  work_grad_output,
                  work_grad_input,
                  kT,
                  kH,
                  kW,
                  1.0f / divide_factor,
                  offsetZ,
                  totalZ);
              totalZ -= 65535;
              offsetZ += 65535;
            }
          });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          input.scalar_type(),
          "avg_pool3d_backward_xpu",
          [&] {
            using accscalar_t = acc_type_device<scalar_t, kXPU>;
            int64_t totalZ = otime * nslices * nbatch;
            int64_t offsetZ = 0;
            while (totalZ > 0) {
              if (kernelsOverlap) {
                avg_pool3d_backward_atomic_template<
                    scalar_t,
                    accscalar_t,
                    int32_t>(
                    work_grad_output,
                    work_grad_input,
                    kT,
                    kH,
                    kW,
                    dT,
                    dH,
                    dW,
                    padT,
                    padH,
                    padW,
                    count_include_pad,
                    offsetZ,
                    totalZ,
                    divisor);
              } else {
                avg_pool3d_backward_template<scalar_t, accscalar_t, int32_t>(
                    work_grad_output,
                    work_grad_input,
                    kT,
                    kH,
                    kW,
                    dT,
                    dH,
                    dW,
                    padT,
                    padH,
                    padW,
                    count_include_pad,
                    offsetZ,
                    totalZ,
                    divisor);
              }

              totalZ -= 65535;
              offsetZ += 65535;
            }
          });
    }
  } else {
    if (dT == 1 && dH == 1 && dW == 1 && padT == 0 && padH == 0 && padW == 0) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          input.scalar_type(),
          "avg_pool3d_backward_stride1_xpu",
          [&] {
            using accscalar_t = acc_type_device<scalar_t, kXPU>;
            int64_t totalZ = itime * nslices * nbatch;
            int64_t offsetZ = 0;

            accscalar_t divide_factor;
            if (divisor) {
              divide_factor = static_cast<accscalar_t>(divisor);
            } else {
              divide_factor = static_cast<accscalar_t>(kT * kH * kW);
            }

            while (totalZ > 0) {
              avg_pool3d_backward_stride1_template<
                  scalar_t,
                  accscalar_t,
                  int64_t>(
                  work_grad_output,
                  work_grad_input,
                  kT,
                  kH,
                  kW,
                  1.0f / divide_factor,
                  offsetZ,
                  totalZ);
              totalZ -= 65535;
              offsetZ += 65535;
            }
          });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          kHalf,
          kBFloat16,
          input.scalar_type(),
          "avg_pool3d_backward_xpu",
          [&] {
            using accscalar_t = acc_type_device<scalar_t, kXPU>;
            int64_t totalZ = otime * nslices * nbatch;
            int64_t offsetZ = 0;
            while (totalZ > 0) {
              if (kernelsOverlap) {
                avg_pool3d_backward_atomic_template<
                    scalar_t,
                    accscalar_t,
                    int64_t>(
                    work_grad_output,
                    work_grad_input,
                    kT,
                    kH,
                    kW,
                    dT,
                    dH,
                    dW,
                    padT,
                    padH,
                    padW,
                    count_include_pad,
                    offsetZ,
                    totalZ,
                    divisor);
              } else {
                avg_pool3d_backward_template<scalar_t, accscalar_t, int64_t>(
                    work_grad_output,
                    work_grad_input,
                    kT,
                    kH,
                    kW,
                    dT,
                    dH,
                    dW,
                    padT,
                    padH,
                    padW,
                    count_include_pad,
                    offsetZ,
                    totalZ,
                    divisor);
              }

              totalZ -= 65535;
              offsetZ += 65535;
            }
          });
    }
  }
}

} // namespace at::native::xpu
#pragma clang diagnostic pop
#pragma GCC diagnostic pop
