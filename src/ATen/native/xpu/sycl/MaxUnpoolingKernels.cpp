#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <comm/MemoryFormat.h>
#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

template <typename T>
inline T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename scalar_t>
struct MaxUnpooling2dForwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto output_ptr = output_data_;
    auto input_ptr = input_data_;
    auto indices_ptr = indices_data_;
    for (int linearIndex = item.get_global_id(0);
         linearIndex < numInputElements_;
         linearIndex += item.get_global_range()[0]) {
      int c = is_channels_last
          ? linearIndex % numChannels_
          : (linearIndex / inputWidth_ / inputHeight_) % numChannels_;
      int n = linearIndex / inputWidth_ / inputHeight_ / numChannels_;
      int maxind = indices_ptr[linearIndex];
      int offset = is_channels_last
          ? n * numChannels_ * outputHeight_ * outputWidth_ + c
          : (n * numChannels_ + c) * outputHeight_ * outputWidth_;
      output_ptr += offset;
      if (is_channels_last_) {
        output_ptr[maxind * numChannels_] = input_ptr[linearIndex];
      } else {
        output_ptr[maxind] = input_ptr[linearIndex];
      }
    }
  };
  MaxUnpooling2dForwardKernelFunctor(
      const int64_t numInputElements,
      const scalar_t* input_data,
      const int64_t* indices_data,
      const int64_t numChannels,
      const int64_t inputHeight,
      const int64_t inputWidth,
      const int64_t outputHeight,
      const int64_t outputWidth,
      scalar_t* output_data,
      const bool is_channels_last)
      : numInputElements_(numInputElements),
        input_data_(input_data),
        indices_data_(indices_data),
        numChannels_(numChannels),
        inputHeight_(inputHeight),
        inputWidth_(inputWidth),
        outputHeight_(outputHeight),
        outputWidth_(outputWidth),
        output_data_(output_data),
        is_channels_last_(is_channels_last) {}

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
  const bool is_channels_last_;
};

template <typename scalar_t>
void max_unpooling2d_forward_kernel(
    const int64_t numInputElements,
    const scalar_t* input,
    const int64_t* indices,
    const int64_t numChannels,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t outputHeight,
    const int64_t outputWidth,
    scalar_t* output,
    const bool is_channels_last) {
  MaxUnpooling2dForwardKernelFunctor<scalar_t> kfn(
      numInputElements,
      input,
      indices,
      numChannels,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      output,
      is_channels_last);

  int64_t group_size = syclMaxWorkGroupSize(kfn);
  int64_t num_groups = CeilDiv(numInputElements, group_size);
  int64_t total_items = num_groups * group_size;

  sycl_kernel_submit(total_items, group_size, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t>
struct MaxUnpooling3dForwardKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto output_ptr = output_data_;
    auto input_ptr = input_data_;
    auto indices_ptr = indices_data_;

    int64_t iColumn = item.get_global_id(0);
    int64_t iRow = item.get_global_id(1);
    int64_t iFrame = (item.get_group()[2] + offsetZ_) % iT_; // input frame/time
    int64_t slice =
        (item.get_group()[2] + offsetZ_) / iT_; // input slice/feature
    if (iRow < iH_ && iColumn < iW_) {
      scalar_t val = input_ptr
          [slice * iT_ * iH_ * iW_ + iFrame * iH_ * iW_ + iRow * iW_ +
           iColumn] /*[slice][iFrame][iRow][iColumn]*/;
      int64_t index = indices_ptr
          [slice * iT_ * iH_ * iW_ + iFrame * iH_ * iW_ + iRow * iW_ +
           iColumn] /*[slice][iFrame][iRow][iColumn]*/;
      output_ptr[slice * oT_ * oH_ * oW_ + index] = val;
    }
  }
  MaxUnpooling3dForwardKernelFunctor(
      scalar_t* input_data,
      int64_t* indices_data,
      scalar_t* output_data,
      const int64_t batchSize,
      const int64_t inputSlices,
      const int64_t iT,
      const int64_t iH,
      const int64_t iW,
      const int64_t oT,
      const int64_t oH,
      const int64_t oW,
      const int64_t offsetZ)
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
  scalar_t* input_data_;
  int64_t* indices_data_;
  scalar_t* output_data_;
  const int64_t batchSize_;
  const int64_t inputSlices_;
  const int64_t iT_;
  const int64_t iH_;
  const int64_t iW_;
  const int64_t oT_;
  const int64_t oH_;
  const int64_t oW_;
  const int64_t offsetZ_;
};

template <typename scalar_t>
void max_unpooling3d_forward_kernel(
    scalar_t* input,
    int64_t* indices,
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
  MaxUnpooling3dForwardKernelFunctor<scalar_t> kfn(
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

  int64_t totalZ = batchSize * inputSlices * iT;
  int64_t num_groups_0 = CeilDiv(iW, (int64_t)32);
  int64_t num_groups_1 = CeilDiv(iH, (int64_t)8);
  sycl::range<3> global_range{
      (size_t)(32 * num_groups_0), (size_t)(8 * num_groups_1), (size_t)totalZ};
  sycl::range<3> local_range{(size_t)32, (size_t)8, (size_t)1};

  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t>
struct MaxUnpooling3dClForwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto output_ptr = output_data_;
    auto input_ptr = input_data_;
    auto indices_ptr = indices_data_;
    for (int linearIndex = item.get_global_id(0);
         linearIndex < numInputElements_;
         linearIndex += item.get_global_range()[0]) {
      int c = linearIndex % numChannels_;
      int n =
          linearIndex / inputDepth_ / inputWidth_ / inputHeight_ / numChannels_;
      int maxind = indices_ptr[linearIndex];
      int offset =
          n * numChannels_ * outputDepth_ * outputHeight_ * outputWidth_ + c;
      output_ptr += offset;
      output_ptr[maxind * numChannels_] = input_ptr[linearIndex];
    }
  }
  MaxUnpooling3dClForwardKernelFunctor(
      const int64_t numInputElements,
      const scalar_t* input_data,
      const int64_t* indices_data,
      const int64_t numChannels,
      const int64_t inputDepth,
      const int64_t inputHeight,
      const int64_t inputWidth,
      const int64_t outputDepth,
      const int64_t outputHeight,
      const int64_t outputWidth,
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
  const int64_t numChannels_;
  const int64_t inputDepth_;
  const int64_t inputHeight_;
  const int64_t inputWidth_;
  const int64_t outputDepth_;
  const int64_t outputHeight_;
  const int64_t outputWidth_;
  scalar_t* output_data_;
};

template <typename scalar_t>
void max_unpooling3d_cl_forward_kernel(
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
  MaxUnpooling3dClForwardKernelFunctor<scalar_t> kfn(
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

  int64_t group_size = syclMaxWorkGroupSize(kfn);
  int64_t num_groups = CeilDiv(numInputElements, group_size);
  int64_t total_items = num_groups * group_size;

  sycl_kernel_submit(total_items, group_size, getCurrentSYCLQueue(), kfn);
}

Tensor& max_unpooling2d_forward_kernel(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size) {
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  auto oheight = output_size[0];
  auto owidth = output_size[1];

  TORCH_CHECK(self_.numel() > 0, "Input must be non-empty tensor");

  TORCH_CHECK(
      (self_.ndimension() == 3 || self_.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor",
      self_.sizes());
  TORCH_CHECK(
      self_.sizes() == indices_.sizes(),
      "Shape of input must match shape of indices");
  TORCH_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (width, height) in output_size");

  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t numBatch = 1;

  int64_t numChannels;
  int64_t inputHeight;
  int64_t inputWidth;

  auto fmt = is_smf_channels_last(self_)
      ? get_cl_tag_by_ndim(self_.ndimension())
      : at::MemoryFormat::Contiguous;
  auto self = self_.contiguous(fmt);
  auto indices = indices_.contiguous(fmt);

  if (self.ndimension() == 4) {
    numBatch = self.size(0);
    dimw++;
    dimh++;
  }
  numChannels = self.size(dimh - 1);
  inputHeight = self.size(dimh);
  inputWidth = self.size(dimw);

  output.resize_({numBatch, numChannels, oheight, owidth}, fmt);

  output.zero_();

  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "max_unpooling2d_forward_xpu",
      ([&] {
        max_unpooling2d_forward_kernel(
            self.numel(),
            self.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(),
            numChannels,
            inputHeight,
            inputWidth,
            oheight,
            owidth,
            output.data_ptr<scalar_t>(),
            is_smf_channels_last(self_));
      }));

  if (self.ndimension() == 3) {
    output.resize_({numChannels, oheight, owidth});
  }
  return output;
}

static void max_unpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "Input to max_unpooling3d should be a 4d or 5d Tensor",
      input.sizes());
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in "
      "output_size");
  TORCH_CHECK(
      stride.size() == 3,
      "There should be exactly three elements (depth, height, width) in "
      "stride");
  TORCH_CHECK(
      padding.size() == 3,
      "There should be exactly three elements (depth, height, width) in "
      "padding");
  TORCH_CHECK(
      input.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");

  TORCH_CHECK(input.numel() > 0, "Input must be non-empty");

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
        "gradOutput and input Tensors should have same number of dimensions "
        "and also the same number of channels/slices");
  }
}

Tensor& max_unpooling3d_forward_kernel(
    Tensor& output,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  max_unpooling3d_shape_check(
      self_, Tensor(), indices_, output_size, stride, padding);

  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  auto fmt = is_smf_channels_last(self_)
      ? get_cl_tag_by_ndim(self_.ndimension())
      : at::MemoryFormat::Contiguous;
  auto self = self_.contiguous(fmt);
  auto indices = indices_.contiguous(fmt);

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
    output.resize_({inputSlices, oT, oH, oW}, fmt);
  } else {
    batchSize = self.size(0);
    inputSlices = self.size(1);
    inputTime = self.size(2);
    inputHeight = self.size(3);
    inputWidth = self.size(4);
    output.resize_({batchSize, inputSlices, oT, oH, oW}, fmt);
  }

  output.zero_();

  if (is_smf_channels_last(self_)) {
    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "max_unpooling3d_cl_forward_xpu",
        ([&] {
          max_unpooling3d_cl_forward_kernel(
              self.numel(),
              self.data_ptr<scalar_t>(),
              indices.data_ptr<int64_t>(),
              inputSlices,
              inputTime,
              inputHeight,
              inputWidth,
              oT,
              oH,
              oW,
              output.data_ptr<scalar_t>());
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

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "max_unpooling3d_forward_xpu",
      ([&] {
        while (totalZ > 0) {
          max_unpooling3d_forward_kernel(
              self.data_ptr<scalar_t>(),
              indices.data_ptr<int64_t>(),
              output.data_ptr<scalar_t>(),
              batchSize,
              inputSlices,
              inputTime,
              inputHeight,
              inputWidth,
              oT,
              oH,
              oW,
              offsetZ);
          totalZ -= 65535;
          offsetZ += 65535;
        }
      }));
  return output;
}

} // namespace at::native::xpu
