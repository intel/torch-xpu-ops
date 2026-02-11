/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/div_rtn.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/DepthwiseConv2dKernels.h>
#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <comm/SYCLContext.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conv_depthwise2d_native.h>
#include <ATen/ops/empty.h>
#endif

namespace at::native::xpu {
#define NUM_THREADS 1024;

template <
    typename scalar_t,
    int ndim,
    template <typename U> class PtrTraits = DefaultPtrTraits>
PackedTensorAccessor32<scalar_t, ndim, PtrTraits> dummy_packed_accessor32() {
  std::array<int64_t, ndim> zeros{};
  return {nullptr, zeros.data(), zeros.data()};
}

template <typename scalar_t, typename index_t>
struct ConvDepthwise2dForwardGenericFunctor {
  void operator()(sycl::nd_item<1> item) const {
    using acc_t = at::acc_type<scalar_t, true>;

    XPU_KERNEL_LOOP_TYPE(item, linearIndex, totalElements, index_t) {
      // calculate n,c,h,w indices, replacing modulos by divide and multiply
      // add, result is same as would be in the code below const int n =
      // linearIndex / batchStride; //batchStride = outputChannels *
      // outputHeight * outputWidth const int c = (linearIndex /
      // channelStride) % outputChannels; //channelStride = outputHeight *
      // outputWidth const int h = (linearIndex / outputWidth) % outputHeight;
      // const int w = linearIndex % outputWidth;

      int indtmp1 = linearIndex / outputWidth;
      const int w = linearIndex - indtmp1 * outputWidth;
      int indtmp2 = indtmp1 / outputHeight;
      const int h = indtmp1 - indtmp2 * outputHeight;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / outputChannels;
      const int c = indtmp1 - indtmp2 * outputChannels;
      const int n = indtmp2;

      int inputChannel = c;
      int inputChannels = outputChannels;
      if (depthwiseMultiplier != 1) {
        inputChannel /= depthwiseMultiplier;
        inputChannels /= depthwiseMultiplier;
      }

      int weightOffset = c * kernelHeight * kernelWidth;

      // By precisely computing the filtering boundaries, we avoid repeating
      // several expensive edge condition checks for every fetched item. If
      // the input element is resident in L1, then the extra branches and
      // comparisons would have been comparable in terms of cycles with the
      // actual data fetch. Therefore computing boundaries ahead of the loop
      // showed significant performance boost.

      int kHmin = 0, kHmax = kernelHeight, kWmin = 0, kWmax = kernelWidth;

      // Top
      int h_in_min = -padHeight + h * strideHeight;
      if (h_in_min < 0) {
        kHmin = -h_in_min / dilationHeight;
        if ((-h_in_min) % dilationHeight > 0) {
          kHmin++;
        }
      }

      // Bottom
      int h_in_max =
          h_in_min + (kernelHeight - 1) * dilationHeight - inputHeight + 1;
      if (h_in_max >= 0) {
        kHmax = kernelHeight - h_in_max / dilationHeight;
        if (h_in_max % dilationHeight > 0) {
          kHmax--;
        }
      }

      // Left
      int w_in_min = -padWidth + w * strideWidth;
      if (w_in_min < 0) {
        kWmin = -w_in_min / dilationWidth;
        if ((-w_in_min) % dilationWidth > 0) {
          kWmin++;
        }
      }

      // Right
      int w_in_max =
          w_in_min + (kernelWidth - 1) * dilationWidth - inputWidth + 1;
      if (w_in_max >= 0) {
        kWmax = kernelWidth - w_in_max / dilationWidth;
        if (w_in_max % dilationWidth > 0) {
          kWmax--;
        }
      }

      acc_t value = biasEnabled ? static_cast<acc_t>(bias.data()[c]) : acc_t(0);
      const index_t offset0 =
          (n * inputChannels + inputChannel) * inputHeight * inputWidth;

      for (int kH = kHmin; kH < kHmax; ++kH) {
        const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
        for (int kW = kWmin; kW < kWmax; ++kW) {
          const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;
          const index_t offset = offset0 + h_in * inputWidth + w_in;
          value +=
              (static_cast<acc_t>(
                   weight.data()[weightOffset + kH * kernelWidth + kW]) *
               static_cast<acc_t>(input.data()[offset]));
        }
      }
      output.data()[linearIndex] = static_cast<scalar_t>(value);
    }
  }

  ConvDepthwise2dForwardGenericFunctor(
      const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> input,
      PackedTensorAccessor32<scalar_t, 4, DefaultPtrTraits> output,
      const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> weight,
      const PackedTensorAccessor32<const scalar_t, 1, DefaultPtrTraits> bias,
      bool biasEnabled,
      index_t totalElements,
      const int outputChannels,
      const int depthwiseMultiplier,
      const int inputWidth,
      const int inputHeight,
      const int outputWidth,
      const int outputHeight,
      const int kernelWidth,
      const int kernelHeight,
      const int strideWidth,
      const int strideHeight,
      const int padWidth,
      const int padHeight,
      const int dilationWidth,
      const int dilationHeight)
      : input(input),
        output(output),
        weight(weight),
        bias(bias),
        biasEnabled(biasEnabled),
        totalElements(totalElements),
        outputChannels(outputChannels),
        depthwiseMultiplier(depthwiseMultiplier),
        inputWidth(inputWidth),
        inputHeight(inputHeight),
        outputWidth(outputWidth),
        outputHeight(outputHeight),
        kernelWidth(kernelWidth),
        kernelHeight(kernelHeight),
        strideWidth(strideWidth),
        strideHeight(strideHeight),
        padWidth(padWidth),
        padHeight(padHeight),
        dilationWidth(dilationWidth),
        dilationHeight(dilationHeight) {}

 private:
  const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> input;
  PackedTensorAccessor32<scalar_t, 4, DefaultPtrTraits> output;
  const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> weight;
  const PackedTensorAccessor32<const scalar_t, 1, DefaultPtrTraits> bias;
  bool biasEnabled;
  index_t totalElements;
  const int outputChannels;
  const int depthwiseMultiplier;
  const int inputWidth;
  const int inputHeight;
  const int outputWidth;
  const int outputHeight;
  const int kernelWidth;
  const int kernelHeight;
  const int strideWidth;
  const int strideHeight;
  const int padWidth;
  const int padHeight;
  const int dilationWidth;
  const int dilationHeight;
};

template <int kSize, typename scalar_t, typename index_t>
struct ConvDepthwise2dForwardFunctor {
  void operator()(sycl::nd_item<1> item) const {
    using acc_t = at::acc_type<scalar_t, true>;
    const int KW_LIMIT = (kSize != 0) ? kSize : kernelWidth;
    const int KH_LIMIT = (kSize != 0) ? kSize : kernelHeight;

    XPU_KERNEL_LOOP_TYPE(item, linearIndex, totalElements, index_t) {
      // calculate n,c,h,w indices, replacing modulos by divide and multiply
      // add, result is same as would be in the code below const int n =
      // linearIndex / batchStride; //batchStride = outputChannels *
      // outputHeight * outputWidth const int c = (linearIndex /
      // channelStride) % outputChannels; //channelStride = outputHeight *
      // outputWidth const int h = (linearIndex / outputWidth) % outputHeight;
      // const int w = linearIndex % outputWidth;

      int indtmp1 = linearIndex / outputWidth;
      const int w = linearIndex - indtmp1 * outputWidth;
      int indtmp2 = indtmp1 / outputHeight;
      const int h = indtmp1 - indtmp2 * outputHeight;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / outputChannels;
      const int c = indtmp1 - indtmp2 * outputChannels;
      const int n = indtmp2;

      int inputChannel = c;
      int inputChannels = outputChannels;
      if (depthwiseMultiplier != 1) {
        inputChannel /= depthwiseMultiplier;
        inputChannels /= depthwiseMultiplier;
      }

      int weightOffset = c * kernelHeight * kernelWidth;

      acc_t value = biasEnabled ? static_cast<acc_t>(bias.data()[c]) : acc_t(0);
      const index_t offset0 =
          (n * inputChannels + inputChannel) * inputHeight * inputWidth;
#pragma unroll
      for (int kH = 0; kH < KH_LIMIT; ++kH) {
#pragma unroll
        for (int kW = 0; kW < KW_LIMIT; ++kW) {
          const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
          const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;

          if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) &&
              (w_in < inputWidth)) {
            const index_t offset = offset0 + h_in * inputWidth + w_in;
            value +=
                (static_cast<acc_t>(weight.data()[weightOffset]) *
                 static_cast<acc_t>(input.data()[offset]));
          }
          ++weightOffset;
        }
      }
      output.data()[linearIndex] = static_cast<scalar_t>(value);
    }
  }
  ConvDepthwise2dForwardFunctor(
      const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> input,
      PackedTensorAccessor32<scalar_t, 4, DefaultPtrTraits> output,
      const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> weight,
      const PackedTensorAccessor32<const scalar_t, 1, DefaultPtrTraits> bias,
      bool biasEnabled,
      index_t totalElements,
      const int outputChannels,
      const int depthwiseMultiplier,
      const int inputWidth,
      const int inputHeight,
      const int outputWidth,
      const int outputHeight,
      const int kernelWidth,
      const int kernelHeight,
      const int strideWidth,
      const int strideHeight,
      const int padWidth,
      const int padHeight,
      const int dilationWidth,
      const int dilationHeight)
      : input(input),
        output(output),
        weight(weight),
        bias(bias),
        biasEnabled(biasEnabled),
        totalElements(totalElements),
        outputChannels(outputChannels),
        depthwiseMultiplier(depthwiseMultiplier),
        inputWidth(inputWidth),
        inputHeight(inputHeight),
        outputWidth(outputWidth),
        outputHeight(outputHeight),
        kernelWidth(kernelWidth),
        kernelHeight(kernelHeight),
        strideWidth(strideWidth),
        strideHeight(strideHeight),
        padWidth(padWidth),
        padHeight(padHeight),
        dilationWidth(dilationWidth),
        dilationHeight(dilationHeight) {}

 private:
  const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> input;
  PackedTensorAccessor32<scalar_t, 4, DefaultPtrTraits> output;
  const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> weight;
  const PackedTensorAccessor32<const scalar_t, 1, DefaultPtrTraits> bias;
  bool biasEnabled;
  index_t totalElements;
  const int outputChannels;
  const int depthwiseMultiplier;
  const int inputWidth;
  const int inputHeight;
  const int outputWidth;
  const int outputHeight;
  const int kernelWidth;
  const int kernelHeight;
  const int strideWidth;
  const int strideHeight;
  const int padWidth;
  const int padHeight;
  const int dilationWidth;
  const int dilationHeight;
};

template <int kSize, int stride, typename scalar_t, typename index_t>
struct ConvDepthwise2dBackwardFunctor {
  void operator()(sycl::nd_item<1> item) const {
    using acc_t = at::acc_type<scalar_t, true>;
    const int KW_LIMIT = (kSize != 0) ? kSize : kernelWidth;
    const int KH_LIMIT = (kSize != 0) ? kSize : kernelHeight;
    const int strideW = (stride != 0) ? stride : strideWidth;
    const int strideH = (stride != 0) ? stride : strideHeight;

    XPU_KERNEL_LOOP_TYPE(item, linearIndex, totalElements, index_t) {
      int indtmp1 = linearIndex / inputWidth;
      const int w = linearIndex - indtmp1 * inputWidth;
      int indtmp2 = indtmp1 / inputHeight;
      const int h = indtmp1 - indtmp2 * inputHeight;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / inputChannels;
      const int c = indtmp1 - indtmp2 * inputChannels;
      const int n = indtmp2;

      acc_t value(0);

      for (int multiplier = 0; multiplier < depthwiseMultiplier; ++multiplier) {
        int och = (c * depthwiseMultiplier) + multiplier;
        int weightOffset = och * kernelHeight * kernelWidth;
        for (int kh = 0; kh < KH_LIMIT; ++kh) {
#pragma unroll
          for (int kw = 0; kw < KW_LIMIT; ++kw) {
            int h_out = h + padHeight - kh * dilationHeight;
            int w_out = w + padWidth - kw * dilationWidth;
            if ((h_out % strideH == 0) && (w_out % strideW == 0)) {
              h_out = h_out / strideH;
              w_out = w_out / strideW;

              if ((h_out >= 0) && (h_out < outputHeight) && (w_out >= 0) &&
                  (w_out < outputWidth)) {
                const int offset =
                    ((n * outputChannels + och) * outputHeight + h_out) *
                        outputWidth +
                    w_out;
                value +=
                    (static_cast<acc_t>(weight.data()[weightOffset]) *
                     static_cast<acc_t>(grad_output.data()[offset]));
              }
            }
            ++weightOffset;
          }
        }
      }
      grad_input.data()[linearIndex] = static_cast<scalar_t>(value);
    }
  }

  ConvDepthwise2dBackwardFunctor(
      const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits>
          grad_output,
      PackedTensorAccessor32<scalar_t, 4, DefaultPtrTraits> grad_input,
      const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> weight,
      index_t totalElements,
      const int inputChannels,
      const int depthwiseMultiplier,
      const int outputChannels,
      const int inputWidth,
      const int inputHeight,
      const int outputWidth,
      const int outputHeight,
      const int kernelWidth,
      const int kernelHeight,
      const int strideWidth,
      const int strideHeight,
      const int padWidth,
      const int padHeight,
      const int dilationWidth,
      const int dilationHeight)
      : grad_output(grad_output),
        grad_input(grad_input),
        weight(weight),
        totalElements(totalElements),
        inputChannels(inputChannels),
        depthwiseMultiplier(depthwiseMultiplier),
        outputChannels(outputChannels),
        inputWidth(inputWidth),
        inputHeight(inputHeight),
        outputWidth(outputWidth),
        outputHeight(outputHeight),
        kernelWidth(kernelWidth),
        kernelHeight(kernelHeight),
        strideWidth(strideWidth),
        strideHeight(strideHeight),
        padWidth(padWidth),
        padHeight(padHeight),
        dilationWidth(dilationWidth),
        dilationHeight(dilationHeight) {}

 private:
  const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> grad_output;
  PackedTensorAccessor32<scalar_t, 4, DefaultPtrTraits> grad_input;
  const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> weight;
  index_t totalElements;
  const int inputChannels;
  const int depthwiseMultiplier;
  const int outputChannels;
  const int inputWidth;
  const int inputHeight;
  const int outputWidth;
  const int outputHeight;
  const int kernelWidth;
  const int kernelHeight;
  const int strideWidth;
  const int strideHeight;
  const int padWidth;
  const int padHeight;
  const int dilationWidth;
  const int dilationHeight;
};

template <typename scalar_t, typename acc_t, typename index_t, int SIMD>
struct ConvDepthwise2dGradWeightFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  SYCL_REQD_SUB_GROUP_SIZE(SIMD) void operator()(
      sycl::nd_item<1> item) const {
    // using acc_t = at::acc_type<scalar_t, true>;
    const int channelStride = kernelWidth * kernelHeight;

    // Each Block is responsible for accumulating over a permutation of
    // (channels x kH x kW), use blockIdx to determine which one
    int bidx = item.get_group(0);
    int kW = bidx % kernelWidth;
    int kH = (bidx / kernelWidth) % kernelHeight;
    int ch = (bidx / channelStride);

    // Need to calculate which input channel is associated with this filter
    // channel
    int inputCh = ch / depthwiseMultiplier;

    acc_t grad(0);

    const int laneId = item.get_local_id(0) % C10_WARP_SIZE;
    const int batch = item.get_local_id(0) / C10_WARP_SIZE;
    const int nwarps = item.get_local_range(0) / C10_WARP_SIZE;
    const int imageElements = outputWidth * outputHeight;
    // Use warp per item.  In the original kernel, a threadblock was used to
    // sum over NHW. Here, we use a warp to sum values over HW dimension, and
    // if batchSize is larger than the number of warps, a warp would loop over
    // remaining batch items (e.g. if there are 8 warps, warp 0 would go over
    // 0-8-16 etc image, warp 1 over 1-9-17 etc). Later in blockReduce, all
    // the warps will be reduced anyway, thus the full reduction will be over
    // NHW, like it should be. That allows to get rid of one modulo operation
    // inside the loop (because n/batchIdx now does not have to be computed
    // through modulo, you are just looping over it), and bring a nice
    // speed-up.
    for (int batchIdx = batch; batchIdx < batchSize; batchIdx += nwarps) {
      // Warp-stride loop over elements in a batch item
      for (index_t idx = laneId; idx < imageElements; idx += C10_WARP_SIZE) {
        // Need to calculate the following: batch position, and offset into
        // the grad_output in height, and width. We can intuit the
        // corresponding position in the input from the other parameters we
        // have
        int go_w_offset = idx % outputWidth;
        int go_h_offset = (idx / outputWidth);

        int i_w_offset =
            (go_w_offset * strideWidth) + (kW * dilationWidth) - padWidth;
        int i_h_offset =
            (go_h_offset * strideHeight) + (kH * dilationHeight) - padHeight;

        if (i_w_offset >= 0 && i_h_offset >= 0 && i_w_offset < inputWidth &&
            i_h_offset < inputHeight) {
          int inputOffset =
              ((batchIdx * inputChannels + inputCh) * inputHeight +
               i_h_offset) *
                  inputWidth +
              i_w_offset;
          int outputOffset =
              ((batchIdx * kernelChannels + ch) * outputHeight) * outputWidth +
              idx;
          grad +=
              (static_cast<acc_t>(input.data()[inputOffset]) *
               static_cast<acc_t>(grad_output.data()[outputOffset]));
        }
      }
    }

    // At this point each thread in the block has a local gradient, which we
    // need to accumulate prior to writing the global value
    acc_t tval = GroupReduceSumWithoutBroadcast<acc_t, SIMD>(item, grad, smem);

    // After reduction, first thread in the block has the gradient, so its
    // responsible for writing it to grad_weight
    if (item.get_local_id(0) == 0) {
      int weightOffset =
          kW + (kernelWidth * kH) + (kernelWidth * kernelHeight * ch);
      grad_weight.data()[weightOffset] = static_cast<scalar_t>(tval);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    smem = sycl_local_acc_t<acc_t>(work_group_size, cgh);
  }

  ConvDepthwise2dGradWeightFunctor(
      const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits>
          grad_output,
      const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> input,
      PackedTensorAccessor32<scalar_t, 4, DefaultPtrTraits> grad_weight,
      const int batchSize,
      const int inputChannels,
      const int kernelChannels,
      const int depthwiseMultiplier,
      const int inputWidth,
      const int inputHeight,
      const int outputWidth,
      const int outputHeight,
      const int kernelWidth,
      const int kernelHeight,
      const int strideWidth,
      const int strideHeight,
      const int padWidth,
      const int padHeight,
      const int dilationWidth,
      const int dilationHeight,
      const int work_group_size)
      : grad_output(grad_output),
        input(input),
        grad_weight(grad_weight),
        batchSize(batchSize),
        inputChannels(inputChannels),
        kernelChannels(kernelChannels),
        depthwiseMultiplier(depthwiseMultiplier),
        inputWidth(inputWidth),
        inputHeight(inputHeight),
        outputWidth(outputWidth),
        outputHeight(outputHeight),
        kernelWidth(kernelWidth),
        kernelHeight(kernelHeight),
        strideWidth(strideWidth),
        strideHeight(strideHeight),
        padWidth(padWidth),
        padHeight(padHeight),
        dilationWidth(dilationWidth),
        dilationHeight(dilationHeight),
        work_group_size(work_group_size) {}

 private:
  const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> grad_output;
  const PackedTensorAccessor32<const scalar_t, 4, DefaultPtrTraits> input;
  PackedTensorAccessor32<scalar_t, 4, DefaultPtrTraits> grad_weight;
  const int batchSize;
  const int inputChannels;
  const int kernelChannels;
  const int depthwiseMultiplier;
  const int inputWidth;
  const int inputHeight;
  const int outputWidth;
  const int outputHeight;
  const int kernelWidth;
  const int kernelHeight;
  const int strideWidth;
  const int strideHeight;
  const int padWidth;
  const int padHeight;
  const int dilationWidth;
  const int dilationHeight;
  const int work_group_size;
  sycl_local_acc_t<acc_t> smem;
};

void conv_depthwise2d_forward_kernel(
    const Tensor& input,
    const Tensor& output,
    const Tensor& weight,
    const Tensor& bias,
    const int kW,
    const int kH,
    const int dW,
    const int dH,
    const int padW,
    const int padH,
    const int dilationW,
    const int dilationH) {
  // Only handle 4D Input Tensors for now
  TORCH_CHECK(input.numel() > 0 && input.dim() == 4);
  TORCH_CHECK(weight.numel() > 0 && weight.dim() == 4);
  TORCH_CHECK(output.is_contiguous());

  auto in_sizes = input.sizes();
  auto w_sizes = weight.sizes();

  // We assume that the input and weight Tensors are shaped properly by
  // the caller, so we verify that here to some extent

  // Weight Tensor is shape (output_channels, 1, kH, kW)
  TORCH_CHECK(w_sizes[1] == 1);

  // Input Tensor is shape (N, input_channels, H, W)
  // We verify that the # of output_channels is a multiple of input_channels
  TORCH_CHECK(w_sizes[0] % in_sizes[1] == 0);

  // Bias has same # of channels as output
  const bool has_bias = bias.defined();
  TORCH_CHECK(!has_bias || (bias.dim() <= 1 && bias.numel() == w_sizes[0]));

  // Following the behavior of other THCUNN functions, we shape the output
  // Tensor ourselves
  int64_t height = in_sizes[2];
  int64_t width = in_sizes[3];
  int64_t outputChannels = w_sizes[0];
  auto out_sizes = conv_output_size(
      in_sizes, weight.sizes(), {padH, padW}, {dH, dW}, {dilationH, dilationW});
  const auto outputWidth = out_sizes[3];
  const auto outputHeight = out_sizes[2];

  resize_output(output, out_sizes);

  int64_t inputChannels = in_sizes[1];
  int64_t depthwiseMultiplier = outputChannels / inputChannels;

  // One thread per output value
  TORCH_CHECK(canUse32BitIndexMath(input) && canUse32BitIndexMath(output));
  int32_t n = output.numel();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "conv_depthwise2d_forward_xpu",
      [&] {
        // Create PackedTensorAccessor
        // Kernel currently relies upon all the Tensors to be contiguous, but
        // we made them contiguous above
        const auto input_a = input.packed_accessor32<const scalar_t, 4>();
        const auto weight_a = weight.packed_accessor32<const scalar_t, 4>();
        const auto output_a = output.packed_accessor32<scalar_t, 4>();
        const auto bias_a = has_bias
            ? bias.packed_accessor32<const scalar_t, 1>()
            : dummy_packed_accessor32<const scalar_t, 1>();
        if (kW == 5 && kH == 5) {
          ConvDepthwise2dForwardFunctor<5, scalar_t, int> kfn(
              input_a,
              output_a,
              weight_a,
              bias_a,
              has_bias,
              n,
              outputChannels,
              depthwiseMultiplier,
              width,
              height,
              outputWidth,
              outputHeight,
              kW,
              kH,
              dW,
              dH,
              padW,
              padH,
              dilationW,
              dilationH);
          int64_t local_range = syclMaxWorkGroupSize(kfn);
          auto global_range = (n - 1) / local_range + 1;
          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        } else if (kW == 3 && kH == 3) {
          ConvDepthwise2dForwardFunctor<3, scalar_t, int> kfn(
              input_a,
              output_a,
              weight_a,
              bias_a,
              has_bias,
              n,
              outputChannels,
              depthwiseMultiplier,
              width,
              height,
              outputWidth,
              outputHeight,
              kW,
              kH,
              dW,
              dH,
              padW,
              padH,
              dilationW,
              dilationH);
          int64_t local_range = syclMaxWorkGroupSize(kfn);
          auto global_range = (n - 1) / local_range + 1;
          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        } else if (kW == 1 && kH == 1) {
          ConvDepthwise2dForwardFunctor<1, scalar_t, int> kfn(
              input_a,
              output_a,
              weight_a,
              bias_a,
              has_bias,
              n,
              outputChannels,
              depthwiseMultiplier,
              width,
              height,
              outputWidth,
              outputHeight,
              kW,
              kH,
              dW,
              dH,
              padW,
              padH,
              dilationW,
              dilationH);
          int64_t local_range = syclMaxWorkGroupSize(kfn);
          auto global_range = (n - 1) / local_range + 1;
          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        } else {
          ConvDepthwise2dForwardGenericFunctor<scalar_t, int> kfn(
              input_a,
              output_a,
              weight_a,
              bias_a,
              has_bias,
              n,
              outputChannels,
              depthwiseMultiplier,
              width,
              height,
              outputWidth,
              outputHeight,
              kW,
              kH,
              dW,
              dH,
              padW,
              padH,
              dilationW,
              dilationH);
          int64_t local_range = syclMaxWorkGroupSize(kfn);
          auto global_range = (n - 1) / local_range + 1;
          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        }
      });
}

void conv_depthwise2d_backward_kernel(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& grad_input,
    const Tensor& weight,
    const int kW,
    const int kH,
    const int dW,
    const int dH,
    const int padW,
    const int padH,
    const int dilationW,
    const int dilationH) {
  // Only handle 4D Input Tensors for now
  TORCH_CHECK(input.numel() > 0 && input.dim() == 4);
  TORCH_CHECK(weight.numel() > 0 && weight.dim() == 4);
  TORCH_CHECK(grad_output.numel() > 0 && grad_output.dim() == 4);

  // Minimal shape checking, as above
  // Same # of elements in batch
  TORCH_CHECK(input.sizes()[0] == grad_output.sizes()[0]);
  // Same # of filters as outputChannels
  TORCH_CHECK(weight.sizes()[0] == grad_output.sizes()[1]);

  // Resize Grainput_a
  auto in_sizes = input.sizes();
  resize_output(grad_input, in_sizes);

  int inputChannels = in_sizes[1];
  int height = in_sizes[2];
  int width = in_sizes[3];

  auto gO_sizes = grad_output.sizes();
  int outputChannels = gO_sizes[1];
  int outputHeight = gO_sizes[2];
  int outputWidth = gO_sizes[3];

  int depthwiseMultiplier = outputChannels / inputChannels;

  // Kernel currently relies upon all the Tensors to be contiguous
  TORCH_CHECK(grad_output.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(grad_input.is_contiguous());

  // One thread per grainput_a value
  TORCH_CHECK(
      canUse32BitIndexMath(grad_input) && canUse32BitIndexMath(grad_output));
  int32_t n = grad_input.numel();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      grad_output.scalar_type(),
      "conv_depthwise2d_backward_xpu",
      [&] {
        auto grad_output_a = grad_output.packed_accessor32<const scalar_t, 4>();
        auto grad_input_a = grad_input.packed_accessor32<scalar_t, 4>();
        auto weight_a = weight.packed_accessor32<const scalar_t, 4>();

        if (kW == 5 && kH == 5) {
          if (dW == 1 && dH == 1) {
            ConvDepthwise2dBackwardFunctor<5, 1, scalar_t, int> kfn(
                grad_output_a,
                grad_input_a,
                weight_a,
                n,
                inputChannels,
                depthwiseMultiplier,
                outputChannels,
                width,
                height,
                outputWidth,
                outputHeight,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
            int64_t local_range = syclMaxWorkGroupSize(kfn);
            auto global_range = (n - 1) / local_range + 1;
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          } else if (dW == 2 && dH == 2) {
            ConvDepthwise2dBackwardFunctor<5, 2, scalar_t, int> kfn(
                grad_output_a,
                grad_input_a,
                weight_a,
                n,
                inputChannels,
                depthwiseMultiplier,
                outputChannels,
                width,
                height,
                outputWidth,
                outputHeight,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
            int64_t local_range = syclMaxWorkGroupSize(kfn);
            auto global_range = (n - 1) / local_range + 1;
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          } else {
            ConvDepthwise2dBackwardFunctor<5, 0, scalar_t, int> kfn(
                grad_output_a,
                grad_input_a,
                weight_a,
                n,
                inputChannels,
                depthwiseMultiplier,
                outputChannels,
                width,
                height,
                outputWidth,
                outputHeight,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
            int64_t local_range = syclMaxWorkGroupSize(kfn);
            auto global_range = (n - 1) / local_range + 1;
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          }
        } else if (kW == 3 && kH == 3) {
          if (dW == 1 && dH == 1) {
            ConvDepthwise2dBackwardFunctor<3, 1, scalar_t, int> kfn(
                grad_output_a,
                grad_input_a,
                weight_a,
                n,
                inputChannels,
                depthwiseMultiplier,
                outputChannels,
                width,
                height,
                outputWidth,
                outputHeight,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
            int64_t local_range = syclMaxWorkGroupSize(kfn);
            auto global_range = (n - 1) / local_range + 1;
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          } else if (dW == 2 && dH == 2) {
            ConvDepthwise2dBackwardFunctor<3, 2, scalar_t, int> kfn(
                grad_output_a,
                grad_input_a,
                weight_a,
                n,
                inputChannels,
                depthwiseMultiplier,
                outputChannels,
                width,
                height,
                outputWidth,
                outputHeight,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
            int64_t local_range = syclMaxWorkGroupSize(kfn);
            auto global_range = (n - 1) / local_range + 1;
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          } else {
            ConvDepthwise2dBackwardFunctor<3, 0, scalar_t, int> kfn(
                grad_output_a,
                grad_input_a,
                weight_a,
                n,
                inputChannels,
                depthwiseMultiplier,
                outputChannels,
                width,
                height,
                outputWidth,
                outputHeight,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
            int64_t local_range = syclMaxWorkGroupSize(kfn);
            auto global_range = (n - 1) / local_range + 1;
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          }
        } else if (kW == 1 && kH == 1) {
          if (dW == 1 && dH == 1) {
            ConvDepthwise2dBackwardFunctor<1, 1, scalar_t, int> kfn(
                grad_output_a,
                grad_input_a,
                weight_a,
                n,
                inputChannels,
                depthwiseMultiplier,
                outputChannels,
                width,
                height,
                outputWidth,
                outputHeight,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
            int64_t local_range = syclMaxWorkGroupSize(kfn);
            auto global_range = (n - 1) / local_range + 1;
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          } else if (dW == 2 && dH == 2) {
            ConvDepthwise2dBackwardFunctor<1, 2, scalar_t, int> kfn(
                grad_output_a,
                grad_input_a,
                weight_a,
                n,
                inputChannels,
                depthwiseMultiplier,
                outputChannels,
                width,
                height,
                outputWidth,
                outputHeight,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
            int64_t local_range = syclMaxWorkGroupSize(kfn);
            auto global_range = (n - 1) / local_range + 1;
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          } else {
            ConvDepthwise2dBackwardFunctor<1, 0, scalar_t, int> kfn(
                grad_output_a,
                grad_input_a,
                weight_a,
                n,
                inputChannels,
                depthwiseMultiplier,
                outputChannels,
                width,
                height,
                outputWidth,
                outputHeight,
                kW,
                kH,
                dW,
                dH,
                padW,
                padH,
                dilationW,
                dilationH);
            int64_t local_range = syclMaxWorkGroupSize(kfn);
            auto global_range = (n - 1) / local_range + 1;
            sycl_kernel_submit(
                global_range * local_range,
                local_range,
                getCurrentSYCLQueue(),
                kfn);
          }
        } else if (dW == 1 && dH == 1) {
          ConvDepthwise2dBackwardFunctor<0, 1, scalar_t, int> kfn(
              grad_output_a,
              grad_input_a,
              weight_a,
              n,
              inputChannels,
              depthwiseMultiplier,
              outputChannels,
              width,
              height,
              outputWidth,
              outputHeight,
              kW,
              kH,
              dW,
              dH,
              padW,
              padH,
              dilationW,
              dilationH);
          int64_t local_range = syclMaxWorkGroupSize(kfn);
          auto global_range = (n - 1) / local_range + 1;
          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        } else if (dW == 2 && dH == 2) {
          ConvDepthwise2dBackwardFunctor<0, 2, scalar_t, int> kfn(
              grad_output_a,
              grad_input_a,
              weight_a,
              n,
              inputChannels,
              depthwiseMultiplier,
              outputChannels,
              width,
              height,
              outputWidth,
              outputHeight,
              kW,
              kH,
              dW,
              dH,
              padW,
              padH,
              dilationW,
              dilationH);
          int64_t local_range = syclMaxWorkGroupSize(kfn);
          auto global_range = (n - 1) / local_range + 1;
          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        } else {
          ConvDepthwise2dBackwardFunctor<0, 0, scalar_t, int> kfn(
              grad_output_a,
              grad_input_a,
              weight_a,
              n,
              inputChannels,
              depthwiseMultiplier,
              outputChannels,
              width,
              height,
              outputWidth,
              outputHeight,
              kW,
              kH,
              dW,
              dH,
              padW,
              padH,
              dilationW,
              dilationH);
          int64_t local_range = syclMaxWorkGroupSize(kfn);
          auto global_range = (n - 1) / local_range + 1;
          sycl_kernel_submit(
              global_range * local_range,
              local_range,
              getCurrentSYCLQueue(),
              kfn);
        }
      });
}

void conv_depthwise2d_grad_weight_kernel(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& grad_weight,
    const int kW,
    const int kH,
    const int dW,
    const int dH,
    const int padW,
    const int padH,
    const int dilationW,
    const int dilationH) {
  // Only handle 4D Input Tensors for now
  TORCH_CHECK(input.numel() > 0 && input.dim() == 4);
  TORCH_CHECK(grad_output.numel() > 0 && grad_output.dim() == 4);

  // Minimal shape checking as above
  // Same # of elements in batch
  TORCH_CHECK(input.sizes()[0] == grad_output.sizes()[0]);

  auto in_sizes = input.sizes();
  int batchSize = in_sizes[0];
  int inputChannels = in_sizes[1];
  int height = in_sizes[2];
  int width = in_sizes[3];

  auto gO_sizes = grad_output.sizes();
  int outputChannels = gO_sizes[1];
  int outputHeight = gO_sizes[2];
  int outputWidth = gO_sizes[3];

  int depthwiseMultiplier = outputChannels / inputChannels;

  resize_output(grad_weight, {outputChannels, 1, kH, kW});

  // Kernel currently relies upon all the Tensors to be contiguous
  TORCH_CHECK(grad_output.is_contiguous());
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(grad_weight.is_contiguous());

  // We parallelize so that each block computes a single value in grad_weight
  TORCH_CHECK(canUse32BitIndexMath(input) && canUse32BitIndexMath(grad_output));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      grad_output.scalar_type(),
      "conv_depthwise2d_grad_weight_xpu",
      [&] {
        const auto grad_output_a =
            grad_output.packed_accessor32<const scalar_t, 4>();
        const auto input_a = input.packed_accessor32<const scalar_t, 4>();
        const auto grad_weight_a = grad_weight.packed_accessor32<scalar_t, 4>();
        using acc_t = at::acc_type<scalar_t, true>;

        using KernelClass =
            ConvDepthwise2dGradWeightFunctor<scalar_t, acc_t, int, 32>;
        int max_wg_size = syclMaxWorkGroupSize<KernelClass>();
        // Make sure we have enough threads to perform the reduction, and use
        // this number to create the shared memory size for the reduction
        int64_t local_range = std::min(batchSize * C10_WARP_SIZE, max_wg_size);
        auto global_range = outputChannels * kH * kW;

        auto kfn = KernelClass(
            grad_output_a,
            input_a,
            grad_weight_a,
            batchSize,
            inputChannels,
            outputChannels,
            depthwiseMultiplier,
            width,
            height,
            outputWidth,
            outputHeight,
            kW,
            kH,
            dW,
            dH,
            padW,
            padH,
            dilationW,
            dilationH,
            local_range);

        // int warp_size = C10_WARP_SIZE;
        // TORCH_INTERNAL_ASSERT(local_range % warp_size == 0);
        // int smem = (local_range / warp_size) * sizeof(acc_t);

        // Crude benchmarks suggest 256 is better than 512 and 1024
        // TODO: Autotune/use better heuristics, improve speed more.
        sycl_kernel_submit(
            global_range * local_range,
            local_range,
            getCurrentSYCLQueue(),
            kfn);
      });
}

} // namespace at::native::xpu
