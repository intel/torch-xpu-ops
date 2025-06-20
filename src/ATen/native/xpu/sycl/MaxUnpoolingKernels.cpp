#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/MaxUnpoolingKernels.h>
#include <comm/MemoryFormat.h>
#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

template <typename scalar_t, typename index_t, bool is_channels_last_>
struct MaxUnpooling2dForwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t outputImageSize = outputHeight_ * outputWidth_;
    auto output = output_data_;
    XPU_KERNEL_LOOP(item, linearIndex, numInputElements_) {
      int c = is_channels_last_
          ? linearIndex % numChannels_
          : (linearIndex / inputWidth_ / inputHeight_) % numChannels_;
      int n = linearIndex / inputWidth_ / inputHeight_ / numChannels_;
      int maxind = indices_data_[linearIndex];
      SYCL_KERNEL_ASSERT(maxind >= 0 && maxind < outputImageSize);
      index_t offset = is_channels_last_
          ? n * numChannels_ * outputHeight_ * outputWidth_ + c
          : (n * numChannels_ + c) * outputHeight_ * outputWidth_;
      output += offset;
      if (is_channels_last_) {
        output[maxind * numChannels_] = input_data_[linearIndex];
      } else {
        output[maxind] = input_data_[linearIndex];
      }
    }
  };
  MaxUnpooling2dForwardKernelFunctor(
      const index_t numInputElements,
      const scalar_t* input_data,
      const int64_t* indices_data,
      const index_t numChannels,
      const index_t inputHeight,
      const index_t inputWidth,
      const index_t outputHeight,
      const index_t outputWidth,
      scalar_t* output_data)
      : numInputElements_(numInputElements),
        input_data_(input_data),
        indices_data_(indices_data),
        numChannels_(numChannels),
        inputHeight_(inputHeight),
        inputWidth_(inputWidth),
        outputHeight_(outputHeight),
        outputWidth_(outputWidth),
        output_data_(output_data) {}

 private:
  const int64_t numInputElements_;
  const scalar_t* input_data_;
  const int64_t* indices_data_;
  const int64_t numChannels_;
  const int64_t inputHeight_;
  const int64_t inputWidth_;
  const int64_t outputHeight_;
  const int64_t outputWidth_;
  scalar_t* output_data_;
};

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
  if (count != 0) {
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
                  auto kfn = MaxUnpooling2dForwardKernelFunctor<
                      scalar_t,
                      index_t,
                      true>(
                      count,
                      self.const_data_ptr<scalar_t>(),
                      indices.const_data_ptr<int64_t>(),
                      numChannels,
                      inputHeight,
                      inputWidth,
                      oheight,
                      owidth,
                      output.mutable_data_ptr<scalar_t>());

                  int64_t group_size = syclMaxWorkItemsPerSubSlice();
                  int64_t num_groups = (count + group_size - 1) / group_size;
                  sycl_kernel_submit(
                      num_groups * group_size,
                      group_size,
                      getCurrentSYCLQueue(),
                      kfn);
                } else {
                  auto kfn = MaxUnpooling2dForwardKernelFunctor<
                      scalar_t,
                      index_t,
                      false>(
                      count,
                      self.const_data_ptr<scalar_t>(),
                      indices.const_data_ptr<int64_t>(),
                      numChannels,
                      inputHeight,
                      inputWidth,
                      oheight,
                      owidth,
                      output.mutable_data_ptr<scalar_t>());
                  int64_t group_size = syclMaxWorkItemsPerSubSlice();
                  int64_t num_groups = (count + group_size - 1) / group_size;
                  sycl_kernel_submit(
                      num_groups * group_size,
                      group_size,
                      getCurrentSYCLQueue(),
                      kfn);
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
struct MaxUnpooling3dForwardKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_ptr = output_data_;
    auto input_ptr = input_data_;
    auto indices_ptr = indices_data_;

    index_t iColumn = item.get_global_id(2);
    index_t iRow = item.get_global_id(1);
    index_t iFrame = (item.get_group()[0] + offsetZ_) % iT_; // input frame/time
    index_t slice =
        (item.get_group()[0] + offsetZ_) / iT_; // input slice/feature
    index_t outputImageSize = oT_ * oH_ * oW_;
    if (iRow < iH_ && iColumn < iW_) {
      scalar_t val = input_ptr
          [slice * iT_ * iH_ * iW_ + iFrame * iH_ * iW_ + iRow * iW_ +
           iColumn] /*[slice][iFrame][iRow][iColumn]*/;
      index_t index = indices_ptr
          [slice * iT_ * iH_ * iW_ + iFrame * iH_ * iW_ + iRow * iW_ +
           iColumn] /*[slice][iFrame][iRow][iColumn]*/;
      SYCL_KERNEL_ASSERT(index >= 0 && index < outputImageSize);
      output_ptr[slice * oT_ * oH_ * oW_ + index] = val;
    }
  }
  MaxUnpooling3dForwardKernelFunctor(
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
      const index_t offsetZ)
      : input_data_(input_data),
        indices_data_(indices_data),
        output_data_(output_data),
        batchSize_(batchSize),
        inputSlices_(inputSlices),
        iT_(iT),
        iH_(iH),
        iW_(iW),
        oT_(oT),
        oH_(oH),
        oW_(oW),
        offsetZ_(offsetZ) {}

 private:
  const scalar_t* input_data_;
  const int64_t* indices_data_;
  scalar_t* output_data_;
  const index_t batchSize_;
  const index_t inputSlices_;
  const index_t iT_;
  const index_t iH_;
  const index_t iW_;
  const index_t oT_;
  const index_t oH_;
  const index_t oW_;
  const index_t offsetZ_;
};

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
  MaxUnpooling3dForwardKernelFunctor<scalar_t, index_t> kfn(
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
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t, typename index_t>
struct MaxUnpooling3dClForwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto output_ptr = output_data_;
    auto input_ptr = input_data_;
    auto indices_ptr = indices_data_;
    for (index_t linearIndex = item.get_global_id(0);
         linearIndex < numInputElements_;
         linearIndex += item.get_global_range()[0]) {
      index_t c = linearIndex % numChannels_;
      index_t n =
          linearIndex / inputDepth_ / inputWidth_ / inputHeight_ / numChannels_;
      index_t maxind = indices_ptr[linearIndex];
      index_t offset =
          n * numChannels_ * outputDepth_ * outputHeight_ * outputWidth_ + c;
      output_ptr += offset;
      output_ptr[maxind * numChannels_] = input_ptr[linearIndex];
    }
  }
  MaxUnpooling3dClForwardKernelFunctor(
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
      scalar_t* output_data)
      : numInputElements_(numInputElements),
        input_data_(input_data),
        indices_data_(indices_data),
        numChannels_(numChannels),
        inputDepth_(inputDepth),
        inputHeight_(inputHeight),
        inputWidth_(inputWidth),
        outputDepth_(outputDepth),
        outputHeight_(outputHeight),
        outputWidth_(outputWidth),
        output_data_(output_data) {}

 private:
  const int64_t numInputElements_;
  const scalar_t* input_data_;
  const int64_t* indices_data_;
  const index_t numChannels_;
  const index_t inputDepth_;
  const index_t inputHeight_;
  const index_t inputWidth_;
  const index_t outputDepth_;
  const index_t outputHeight_;
  const index_t outputWidth_;
  scalar_t* output_data_;
};

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
  MaxUnpooling3dClForwardKernelFunctor<scalar_t, index_t> kfn(
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

  int64_t group_size = syclMaxWorkItemsPerSubSlice();
  int64_t num_groups = (numInputElements + group_size - 1) / group_size;
  int64_t total_items = num_groups * group_size;
  sycl_kernel_submit(total_items, group_size, getCurrentSYCLQueue(), kfn);
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
    if (oT != gradOutput.size(dimt) || oH != gradOutput.size(dimh) ||
        oW != gradOutput.size(dimw)) {
      AT_ERROR(
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
    }
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
