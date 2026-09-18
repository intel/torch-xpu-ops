/*
 * Copyright 2020-2026 Intel Corporation
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

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/MaxUnpoolingKernels.h>
#include <comm/MemoryFormat.h>
#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

template <typename scalar_t, typename index_t, bool is_channels_last>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void max_unpooling_2d_forward_kernel_impl(
    const index_t numInputElements,
    const scalar_t* input_data,
    const int64_t* indices_data,
    const index_t numChannels,
    const index_t inputHeight,
    const index_t inputWidth,
    const index_t outputHeight,
    const index_t outputWidth,
    scalar_t* output_data) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  int64_t outputImageSize = outputHeight * outputWidth;
  XPU_KERNEL_LOOP(item, linearIndex, numInputElements) {
    int c = is_channels_last
        ? linearIndex % numChannels
        : (linearIndex / inputWidth / inputHeight) % numChannels;
    int n = linearIndex / inputWidth / inputHeight / numChannels;
    int maxind = indices_data[linearIndex];
    SYCL_KERNEL_ASSERT(maxind >= 0 && maxind < outputImageSize);
    index_t offset = is_channels_last
        ? n * numChannels * outputHeight * outputWidth + c
        : (n * numChannels + c) * outputHeight * outputWidth;
    scalar_t* out = output_data + offset;
    if constexpr (is_channels_last) {
      out[maxind * numChannels] = input_data[linearIndex];
    } else {
      out[maxind] = input_data[linearIndex];
    }
  }
}

Tensor& max_unpooling2d_forward_kernel(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size) {
  at::globalContext().alertNotDeterministic("max_unpooling2d_forward_out");

  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64 but got: ",
      indices_.scalar_type());
  auto oheight = output_size[0];
  auto owidth = output_size[1];

  TensorArg output_arg{output, "output", 1}, self_arg{self_, "self_", 2},
      indices_arg{indices_, "indices_", 3};
  checkAllSameGPU(
      "max_unpooling2d_forward_out_xpu", {output_arg, self_arg, indices_arg});

  for (int64_t i = 1; i < self_.ndimension(); ++i) {
    TORCH_CHECK(
        self_.size(i) > 0,
        "max_unpooling2d_forward_out_xpu(): ",
        "Expected input to have non-zero size for non-batch dimensions, but got ",
        self_.sizes(),
        " with dimension ",
        i,
        " being empty.");
  }

  TORCH_CHECK(
      (self_.ndimension() == 3 || self_.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor, but got tensor with dimension: ",
      self_.ndimension());
  TORCH_CHECK(
      self_.sizes() == indices_.sizes(),
      "Expected shape of indices to be: ",
      self_.sizes(),
      " but got: ",
      indices_.sizes());
  TORCH_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size, but got ",
      output_size.size(),
      " elements.");

  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t numBatch = 1;

  int64_t numChannels;
  int64_t inputHeight;
  int64_t inputWidth;

  auto memory_format = self_.suggest_memory_format();
  auto self = self_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  if (self.ndimension() == 4) {
    numBatch = self.size(0);
    dimw++;
    dimh++;
  }
  numChannels = self.size(dimh - 1);
  inputHeight = self.size(dimh);
  inputWidth = self.size(dimw);

  output.resize_({numBatch, numChannels, oheight, owidth}, memory_format);
  output.zero_();

  auto count = self.numel();
  if (count != 0 && oheight != 0 && owidth != 0) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "max_unpooling2d_forward_xpu",
        ([&] {
          AT_DISPATCH_INDEX_TYPES(
              at::native::canUse32BitIndexMath(output, INT_MAX)
                  ? ScalarType::Int
                  : ScalarType::Long,
              "max_unpooling2d_forward_xpu",
              [&] {
                if (is_channels_last(memory_format)) {
                  int64_t group_size = syclMaxWorkItemsPerSubSlice();
                  int64_t num_groups =
                      xpuKernelLoopGroupRange(count, group_size);
                  sycl_kernel_submit<max_unpooling_2d_forward_kernel_impl<
                      scalar_t,
                      index_t,
                      true>>(
                      num_groups * group_size,
                      group_size,
                      getCurrentSYCLQueue(),
                      0,
                      count,
                      self.const_data_ptr<scalar_t>(),
                      indices.const_data_ptr<int64_t>(),
                      numChannels,
                      inputHeight,
                      inputWidth,
                      oheight,
                      owidth,
                      output.mutable_data_ptr<scalar_t>());
                } else {
                  int64_t group_size = syclMaxWorkItemsPerSubSlice();
                  int64_t num_groups =
                      xpuKernelLoopGroupRange(count, group_size);

                  sycl_kernel_submit<max_unpooling_2d_forward_kernel_impl<
                      scalar_t,
                      index_t,
                      false>>(
                      num_groups * group_size,
                      group_size,
                      getCurrentSYCLQueue(),
                      0,
                      count,
                      self.const_data_ptr<scalar_t>(),
                      indices.const_data_ptr<int64_t>(),
                      numChannels,
                      inputHeight,
                      inputWidth,
                      oheight,
                      owidth,
                      output.mutable_data_ptr<scalar_t>());
                }
              });
        }));
  }
  if (self.ndimension() == 3) {
    output.resize_({numChannels, oheight, owidth});
  }
  return output;
}

template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<3>))
void max_unpooling_3d_forward_kernel_impl(
    const scalar_t* input_data,
    const int64_t* indices_data,
    scalar_t* output_data,
    const index_t batchSize,
    const index_t inputSlices,
    const index_t iT,
    const index_t iH,
    const index_t iW,
    const index_t oT,
    const index_t oH,
    const index_t oW,
    const index_t offsetZ) {
  auto item = syclext::this_work_item::get_nd_item<3>();
  auto output_ptr = output_data;
  auto input_ptr = input_data;
  auto indices_ptr = indices_data;

  index_t iColumn = item.get_global_id(2);
  index_t iRow = item.get_global_id(1);
  index_t iFrame = (item.get_group()[0] + offsetZ) % iT; // input frame/time
  index_t slice = (item.get_group()[0] + offsetZ) / iT; // input slice/feature
  index_t outputImageSize = oT * oH * oW;
  if (iRow < iH && iColumn < iW) {
    scalar_t val = input_ptr
        [slice * iT * iH * iW + iFrame * iH * iW + iRow * iW +
         iColumn] /*[slice][iFrame][iRow][iColumn]*/;
    index_t index = indices_ptr
        [slice * iT * iH * iW + iFrame * iH * iW + iRow * iW +
         iColumn] /*[slice][iFrame][iRow][iColumn]*/;
    SYCL_KERNEL_ASSERT(index >= 0 && index < outputImageSize);
    output_ptr[slice * oT * oH * oW + index] = val;
  }
}

template <typename scalar_t, typename index_t>
void max_unpooling3d_forward_template(
    const scalar_t* input,
    const int64_t* indices,
    scalar_t* output,
    const int64_t batchSize,
    const int64_t inputSlices,
    const int64_t iT,
    const int64_t iH,
    const int64_t iW,
    const int64_t oT,
    const int64_t oH,
    const int64_t oW,
    const int64_t offsetZ) {
  int64_t work_group_size_w = 32;
  int64_t work_group_size_h = syclMaxWorkItemsPerSubSlice() / work_group_size_w;
  int64_t total_t = batchSize * inputSlices * iT;
  // int64_t num_groups_w = CeilDiv(iW, work_group_size_w);
  // int64_t num_groups_h = CeilDiv(iH, work_group_size_h);
  int64_t num_groups_w = (iW + work_group_size_w - 1) / work_group_size_w;
  int64_t num_groups_h = (iH + work_group_size_h - 1) / work_group_size_h;

  sycl::range<3> local_range{
      (size_t)1, (size_t)work_group_size_h, (size_t)work_group_size_w};
  sycl::range<3> global_range{
      (size_t)total_t,
      (size_t)(work_group_size_h * num_groups_h),
      (size_t)(work_group_size_w * num_groups_w)};
  sycl_kernel_submit<max_unpooling_3d_forward_kernel_impl<scalar_t, index_t>>(
      global_range,
      local_range,
      getCurrentSYCLQueue(),
      0,
      input,
      indices,
      output,
      batchSize,
      inputSlices,
      iT,
      iH,
      iW,
      oT,
      oH,
      oW,
      offsetZ);
}

template <typename scalar_t, typename index_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void max_unpooling_3d_cl_forward_kernel_impl(
    const int64_t numInputElements,
    const scalar_t* input_data,
    const int64_t* indices_data,
    const index_t numChannels,
    const index_t inputDepth,
    const index_t inputHeight,
    const index_t inputWidth,
    const index_t outputDepth,
    const index_t outputHeight,
    const index_t outputWidth,
    scalar_t* output_data) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  auto input_ptr = input_data;
  auto indices_ptr = indices_data;
  for (index_t linearIndex = item.get_global_id(0);
       linearIndex < numInputElements;
       linearIndex += item.get_global_range()[0]) {
    index_t c = linearIndex % numChannels;
    index_t n =
        linearIndex / inputDepth / inputWidth / inputHeight / numChannels;
    index_t maxind = indices_ptr[linearIndex];
    index_t offset =
        n * numChannels * outputDepth * outputHeight * outputWidth + c;
    scalar_t* out = output_data + offset;
    out[maxind * numChannels] = input_ptr[linearIndex];
  }
}

template <typename scalar_t, typename index_t>
void max_unpooling3d_cl_forward_template(
    const int64_t numInputElements,
    const scalar_t* input,
    const int64_t* indices,
    const int64_t numChannels,
    const int64_t inputDepth,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t outputDepth,
    const int64_t outputHeight,
    const int64_t outputWidth,
    scalar_t* output) {
  int64_t group_size = syclMaxWorkItemsPerSubSlice();
  int64_t num_groups = xpuKernelLoopGroupRange(numInputElements, group_size);
  int64_t total_items = num_groups * group_size;
  sycl_kernel_submit<
      max_unpooling_3d_cl_forward_kernel_impl<scalar_t, index_t>>(
      total_items,
      group_size,
      getCurrentSYCLQueue(),
      0,
      numInputElements,
      input,
      indices,
      numChannels,
      inputDepth,
      inputHeight,
      inputWidth,
      outputDepth,
      outputHeight,
      outputWidth,
      output);
}

static void max_unpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const char* fn_name) {
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64 but got: ",
      indices.scalar_type());
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "Input to max_unpooling3d should be a 4d or 5d Tensor, but got a tensor with dim ",
      input.ndimension());
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size, but got ",
      output_size.size(),
      " elements.");
  TORCH_CHECK(
      stride.size() == 3,
      "There should be exactly three elements (depth, height, width) in stride, but got: ",
      stride.size(),
      " elements.");
  TORCH_CHECK(
      padding.size() == 3,
      "There should be exactly three elements (depth, height, width) in padding, but got: ",
      padding.size(),
      " elements.");
  TORCH_CHECK(
      input.sizes() == indices.sizes(),
      "Expected shape of indices to be: ",
      input.sizes(),
      " but got: ",
      indices.sizes());

  for (int64_t i = 1; i < input.ndimension(); ++i) {
    TORCH_CHECK(
        input.size(i) > 0,
        fn_name,
        ": Expected input to have non-zero size for non-batch dimensions, but got ",
        input.sizes(),
        " with dimension ",
        i,
        " being empty.");
  }

  TORCH_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "strides should be greater than zero, but got stride: ",
      stride);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  if (input.ndimension() == 5) {
    dimw++;
    dimh++;
    dimt++;
    dimn++;
  }

  int nslices = input.size(dimn);

  if (gradOutput.defined()) {
    TORCH_CHECK(
        oT == gradOutput.size(dimt) && oH == gradOutput.size(dimh) &&
            oW == gradOutput.size(dimw),
        "Inconsistent gradOutput size. oT= ",
        oT,
        ", oH= ",
        oH,
        ", oW= ",
        oW,
        ". gradOutput: ",
        gradOutput.size(dimt),
        "x",
        gradOutput.size(dimh),
        "x",
        gradOutput.size(dimw));
    TORCH_CHECK(
        gradOutput.ndimension() == input.ndimension() &&
            gradOutput.size(dimn) == nslices,
        "gradOutput and input Tensors should have same number of dimensions and also the same number of channels/slices");
  }
}

Tensor& max_unpooling3d_forward_kernel(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  at::globalContext().alertNotDeterministic("max_unpooling3d_forward_out");
  max_unpooling3d_shape_check(
      self_,
      Tensor(),
      indices_,
      output_size,
      stride,
      padding,
      "max_unpooling3d_forward_out_xpu()");

  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  TensorArg output_arg{output, "output", 1}, self_arg{self_, "self_", 2},
      indices_arg{indices_, "indices_", 3};
  checkAllSameGPU(
      "max_unpooling3d_forward_out_xpu", {output_arg, self_arg, indices_arg});
  auto memory_format = self_.suggest_memory_format();
  auto self = self_.contiguous(memory_format);
  auto indices = indices_.contiguous(memory_format);

  int64_t batchSize;
  int64_t inputSlices;
  int64_t inputTime;
  int64_t inputHeight;
  int64_t inputWidth;

  if (self.ndimension() == 4) {
    batchSize = 1;
    inputSlices = self.size(0);
    inputTime = self.size(1);
    inputHeight = self.size(2);
    inputWidth = self.size(3);
    output.resize_({inputSlices, oT, oH, oW}, memory_format);
  } else {
    batchSize = self.size(0);
    inputSlices = self.size(1);
    inputTime = self.size(2);
    inputHeight = self.size(3);
    inputWidth = self.size(4);
    output.resize_({batchSize, inputSlices, oT, oH, oW}, memory_format);
  }

  output.zero_();
  if (is_channels_last(memory_format)) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "max_unpooling3d_forward_xpu",
        ([&] {
          AT_DISPATCH_INDEX_TYPES(
              at::native::canUse32BitIndexMath(output, INT_MAX)
                  ? ScalarType::Int
                  : ScalarType::Long,
              "max_unpooling3d_forward_xpu",
              [&] {
                max_unpooling3d_cl_forward_template<scalar_t, index_t>(
                    self.numel(),
                    self.const_data_ptr<scalar_t>(),
                    indices.const_data_ptr<int64_t>(),
                    inputSlices,
                    inputTime,
                    inputHeight,
                    inputWidth,
                    oT,
                    oH,
                    oW,
                    output.mutable_data_ptr<scalar_t>());
              });
        }));

    return output;
  }
  // Collapse batch and feature dimensions if needed
  if (self.ndimension() == 5) {
    self = self.reshape(
        {self.size(0) * self.size(1),
         self.size(2),
         self.size(3),
         self.size(4)});
    indices = indices.reshape(
        {indices.size(0) * indices.size(1),
         indices.size(2),
         indices.size(3),
         indices.size(4)});
  }

  if (self.numel() == 0) {
    return output;
  }

  if (oT == 0 || oH == 0 || oW == 0) {
    return output;
  }

  int offsetZ = 0;
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "max_unpooling3d_forward_xpu",
      ([&] {
        AT_DISPATCH_INDEX_TYPES(
            at::native::canUse32BitIndexMath(output, INT_MAX)
                ? ScalarType::Int
                : ScalarType::Long,
            "max_unpooling3d_forward_xpu",
            [&] {
              max_unpooling3d_forward_template<scalar_t, index_t>(
                  self.const_data_ptr<scalar_t>(),
                  indices.const_data_ptr<int64_t>(),
                  output.mutable_data_ptr<scalar_t>(),
                  batchSize,
                  inputSlices,
                  inputTime,
                  inputHeight,
                  inputWidth,
                  oT,
                  oH,
                  oW,
                  offsetZ);
            });
      }));
  return output;
}

} // namespace at::native::xpu
