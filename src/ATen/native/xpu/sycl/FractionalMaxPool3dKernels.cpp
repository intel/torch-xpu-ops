#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NumericUtils.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/MemoryFormat.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/FractionalMaxPool3dKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename accscalar_t>
inline int64_t get_intervals(
    accscalar_t sample,
    int64_t index,
    int64_t inputSize,
    int64_t outputSize,
    int64_t poolSize) {
  accscalar_t alpha = static_cast<accscalar_t>(inputSize - poolSize) /
      static_cast<accscalar_t>(outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return static_cast<int64_t>((index + sample) * alpha) -
        static_cast<int64_t>(sample * alpha);
  }
}

template <typename scalar_t, typename accscalar_t>
struct FractionalMaxPool3dOutFrameCfFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto input_ptr = input_data_;
    auto output_ptr = output_data_;
    auto indices_ptr = indices_data_;
    auto samples_ptr = samples_data_;

    int ourOutputPoint = item.get_global_id()[2];
    int batch = item.get_group()[0];
    int plane = item.get_group()[1];

    if (ourOutputPoint < outputPlaneSize_) {
      int64_t outputT = ourOutputPoint / (outputSizeH_ * outputSizeW_);
      int64_t outputH = (ourOutputPoint / outputSizeW_) % outputSizeH_;
      int64_t outputW = ourOutputPoint % outputSizeW_;

      int64_t poolT = get_intervals<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(
              samples_ptr
                  [batch * numPlane_ * 3 + plane * 3] /*[batch][plane][0]*/),
          outputT,
          inputSizeT_,
          outputSizeT_,
          poolSizeT_);
      int64_t poolH = get_intervals<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(samples_ptr
                                       [batch * numPlane_ * 3 + plane * 3 +
                                        1] /*[batch][plane][1]*/),
          outputH,
          inputSizeH_,
          outputSizeH_,
          poolSizeH_);
      int64_t poolW = get_intervals<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(samples_ptr
                                       [batch * numPlane_ * 3 + plane * 3 +
                                        2] /*[batch][plane][2]*/),
          outputW,
          inputSizeW_,
          outputSizeW_,
          poolSizeW_);

      scalar_t maxVal = std::numeric_limits<scalar_t>::lowest();
      int64_t maxIndex = -1;

      for (int64_t t = poolT; t < poolT + poolSizeT_; ++t) {
        for (int64_t h = poolH; h < poolH + poolSizeH_; ++h) {
          for (int64_t w = poolW; w < poolW + poolSizeW_; ++w) {
            int64_t load_offset = batch * ibatch_stride_ +
                plane * iplane_stride_ + t * iT_stride_ + h * inputSizeW_ + w;
            scalar_t val = input_ptr[load_offset] /*[batch][plane][t][h][w]*/;
            if (val > maxVal) {
              maxIndex = t * inputSizeH_ * inputSizeW_ + h * inputSizeW_ + w;
              maxVal = val;
            }
          }
        }
      }

      int64_t store_offset = batch * obatch_stride_ + plane * oplane_stride_ +
          outputT * oT_stride_ + outputH * outputSizeW_ + outputW;
      indices_ptr[store_offset] /*[batch][plane][outputT][outputH][outputW]*/
          = maxIndex;
      output_ptr[store_offset] /*[batch][plane][outputT][outputH][outputW]*/
          = maxVal;
    }
  }
  FractionalMaxPool3dOutFrameCfFunctor(
      scalar_t* output_data,
      int64_t* indices_data,
      scalar_t* input_data,
      scalar_t* samples_data,
      int numBatch,
      int numPlane,
      int inputSizeT,
      int inputSizeH,
      int inputSizeW,
      int outputSizeT,
      int outputSizeH,
      int outputSizeW,
      int poolSizeT,
      int poolSizeH,
      int poolSizeW,
      int outputPlaneSize,
      int64_t iT_stride,
      int64_t iplane_stride,
      int64_t ibatch_stride,
      int64_t oT_stride,
      int64_t oplane_stride,
      int64_t obatch_stride)
      : output_data_(output_data),
        indices_data_(indices_data),
        input_data_(input_data),
        samples_data_(samples_data),
        numBatch_(numBatch),
        numPlane_(numPlane),
        inputSizeT_(inputSizeT),
        inputSizeH_(inputSizeH),
        inputSizeW_(inputSizeW),
        outputSizeT_(outputSizeT),
        outputSizeH_(outputSizeH),
        outputSizeW_(outputSizeW),
        poolSizeT_(poolSizeT),
        poolSizeH_(poolSizeH),
        poolSizeW_(poolSizeW),
        outputPlaneSize_(outputPlaneSize),
        iT_stride_(iT_stride),
        iplane_stride_(iplane_stride),
        ibatch_stride_(ibatch_stride),
        oT_stride_(oT_stride),
        oplane_stride_(oplane_stride),
        obatch_stride_(obatch_stride) {}

 private:
  scalar_t* output_data_;
  int64_t* indices_data_;
  scalar_t* input_data_;
  scalar_t* samples_data_;
  int numBatch_;
  int numPlane_;
  int inputSizeT_;
  int inputSizeH_;
  int inputSizeW_;
  int outputSizeT_;
  int outputSizeH_;
  int outputSizeW_;
  int poolSizeT_;
  int poolSizeH_;
  int poolSizeW_;
  int outputPlaneSize_;
  int64_t iT_stride_;
  int64_t iplane_stride_;
  int64_t ibatch_stride_;
  int64_t oT_stride_;
  int64_t oplane_stride_;
  int64_t obatch_stride_;
};

template <typename scalar_t>
void fractional_max_pool3d_out_frame_cf(
    scalar_t* output,
    int64_t* indices,
    scalar_t* input,
    scalar_t* samples,
    int numBatch,
    int numPlane,
    int inputSizeT,
    int inputSizeH,
    int inputSizeW,
    int outputSizeT,
    int outputSizeH,
    int outputSizeW,
    int poolSizeT,
    int poolSizeH,
    int poolSizeW) {
  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  using KernelClass =
      FractionalMaxPool3dOutFrameCfFunctor<scalar_t, accscalar_t>;
  auto dev_id = getDeviceIndexOfCurrentQueue();
  int64_t max_wg_size = syclMaxWorkGroupSize<KernelClass>(dev_id);
  int outputPlaneSize = outputSizeH * outputSizeW * outputSizeT;
  // input stride for NCTHW data
  int64_t iT_stride = inputSizeH * inputSizeW;
  int64_t iplane_stride = inputSizeT * iT_stride;
  int64_t ibatch_stride = numPlane * iplane_stride;
  // output stride for NCTHW data
  int64_t oT_stride = outputSizeH * outputSizeW;
  int64_t oplane_stride = outputSizeT * oT_stride;
  int64_t obatch_stride = numPlane * oplane_stride;

  int work_group_size =
      outputPlaneSize > max_wg_size ? max_wg_size : outputPlaneSize;
  int work_group_num =
      (outputPlaneSize + work_group_size - 1) / work_group_size;

  auto kfn = KernelClass(
      output,
      indices,
      input,
      samples,
      numBatch,
      numPlane,
      inputSizeT,
      inputSizeH,
      inputSizeW,
      outputSizeT,
      outputSizeH,
      outputSizeW,
      poolSizeT,
      poolSizeH,
      poolSizeW,
      outputPlaneSize,
      iT_stride,
      iplane_stride,
      ibatch_stride,
      oT_stride,
      oplane_stride,
      obatch_stride);

  sycl::range<3> local_range{(size_t)1, (size_t)1, (size_t)work_group_size};
  sycl::range<3> global_range{
      (size_t)numBatch,
      (size_t)numPlane,
      (size_t)(work_group_size * work_group_num)};
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t, typename accscalar_t>
struct FractionalMaxPool3dOutFrameClFunctor {
  void operator()(sycl::nd_item<3> item) const {
    auto input_ptr = input_data_;
    auto output_ptr = output_data_;
    auto indices_ptr = indices_data_;
    auto samples_ptr = samples_data_;

    int outputIndex = item.get_global_id()[2];

    if (outputIndex < outputSize_) {
      int batch = item.get_group()[0];
      int outputT = item.get_group()[1];
      int outputH = outputIndex / numPlane_ / outputSizeW_ % outputSizeH_;
      int outputW = outputIndex / numPlane_ % outputSizeW_;
      int plane = outputIndex % numPlane_;
      int64_t poolT = get_intervals<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(
              samples_ptr
                  [batch * numPlane_ * 3 + plane * 3] /*[batch][plane][0]*/),
          outputT,
          inputSizeT_,
          outputSizeT_,
          poolSizeT_);
      int64_t poolH = get_intervals<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(samples_ptr
                                       [batch * numPlane_ * 3 + plane * 3 +
                                        1] /*[batch][plane][1]*/),
          outputH,
          inputSizeH_,
          outputSizeH_,
          poolSizeH_);
      int64_t poolW = get_intervals<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(samples_ptr
                                       [batch * numPlane_ * 3 + plane * 3 +
                                        2] /*[batch][plane][2]*/),
          outputW,
          inputSizeW_,
          outputSizeW_,
          poolSizeW_);

      scalar_t maxVal = std::numeric_limits<scalar_t>::lowest();
      int64_t maxIndex = -1;

      for (int64_t t = poolT; t < poolT + poolSizeT_; ++t) {
        for (int64_t h = poolH; h < poolH + poolSizeH_; ++h) {
          for (int64_t w = poolW; w < poolW + poolSizeW_; ++w) {
            int64_t load_offset = batch * iBatch_stride_ + t * iT_stride_ +
                h * iH_stride_ + w * numPlane_ + plane;
            scalar_t val = input_ptr[load_offset] /*[batch][plane][t][h][w]*/;
            if (val > maxVal) {
              maxIndex = t * inputSizeH_ * inputSizeW_ + h * inputSizeW_ + w;
              maxVal = val;
            }
          }
        }
      }
      int64_t store_offset = batch * oBatch_stride_ + outputT * oT_stride_ +
          outputH * oH_stride_ + outputW * numPlane_ + plane;
      indices_ptr[store_offset] /*[batch][plane][outputT][outputH][outputW]*/
          = maxIndex;
      output_ptr[store_offset] /*[batch][plane][outputT][outputH][outputW]*/
          = maxVal;
    }
  }
  FractionalMaxPool3dOutFrameClFunctor(
      scalar_t* output_data,
      int64_t* indices_data,
      scalar_t* input_data,
      scalar_t* samples_data,
      int numBatch,
      int numPlane,
      int inputSizeT,
      int inputSizeH,
      int inputSizeW,
      int outputSizeT,
      int outputSizeH,
      int outputSizeW,
      int poolSizeT,
      int poolSizeH,
      int poolSizeW,
      int outputSize,
      int64_t iH_stride,
      int64_t iT_stride,
      int64_t iBatch_stride,
      int64_t oH_stride,
      int64_t oT_stride,
      int64_t oBatch_stride)
      : output_data_(output_data),
        indices_data_(indices_data),
        input_data_(input_data),
        samples_data_(samples_data),
        numBatch_(numBatch),
        numPlane_(numPlane),
        inputSizeT_(inputSizeT),
        inputSizeH_(inputSizeH),
        inputSizeW_(inputSizeW),
        outputSizeT_(outputSizeT),
        outputSizeH_(outputSizeH),
        outputSizeW_(outputSizeW),
        poolSizeT_(poolSizeT),
        poolSizeH_(poolSizeH),
        poolSizeW_(poolSizeW),
        outputSize_(outputSize),
        iH_stride_(iH_stride),
        iT_stride_(iT_stride),
        iBatch_stride_(iBatch_stride),
        oH_stride_(oH_stride),
        oT_stride_(oT_stride),
        oBatch_stride_(oBatch_stride) {}

 private:
  scalar_t* output_data_;
  int64_t* indices_data_;
  scalar_t* input_data_;
  scalar_t* samples_data_;
  int numBatch_;
  int numPlane_;
  int inputSizeT_;
  int inputSizeH_;
  int inputSizeW_;
  int outputSizeT_;
  int outputSizeH_;
  int outputSizeW_;
  int poolSizeT_;
  int poolSizeH_;
  int poolSizeW_;
  int outputSize_;
  int64_t iH_stride_;
  int64_t iT_stride_;
  int64_t iBatch_stride_;
  int64_t oH_stride_;
  int64_t oT_stride_;
  int64_t oBatch_stride_;
};

template <typename scalar_t>
void fractional_max_pool3d_out_frame_cl(
    scalar_t* output,
    int64_t* indices,
    scalar_t* input,
    scalar_t* samples,
    int numBatch,
    int numPlane,
    int inputSizeT,
    int inputSizeH,
    int inputSizeW,
    int outputSizeT,
    int outputSizeH,
    int outputSizeW,
    int poolSizeT,
    int poolSizeH,
    int poolSizeW) {
  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  using KernelClass =
      FractionalMaxPool3dOutFrameClFunctor<scalar_t, accscalar_t>;
  auto dev_id = getDeviceIndexOfCurrentQueue();
  int64_t max_wg_size = syclMaxWorkGroupSize<KernelClass>(dev_id);
  int outputSize = outputSizeH * outputSizeW * numPlane;
  // input stride for NTHWC data
  int64_t iH_stride = inputSizeW * numPlane;
  int64_t iT_stride = inputSizeH * iH_stride;
  // iBatch_stride = inputSizeT * inputSizeH * inputSizeW * numPlane
  int64_t iBatch_stride = inputSizeT * iT_stride;

  // output stride for NTHWC data
  int64_t oH_stride = outputSizeW * numPlane;
  int64_t oT_stride = outputSizeH * oH_stride;
  // oBatch_stride = outputSizeT * outputSizeH * outputSizeW * numPlane
  int64_t oBatch_stride = outputSizeT * oT_stride;

  int work_group_size = outputSize > max_wg_size ? max_wg_size : outputSize;
  int work_group_num = (outputSize + work_group_size - 1) / work_group_size;

  auto input_data = input;
  auto output_data = output;
  auto indices_data = indices;
  auto samples_data = samples;

  auto kfn = KernelClass(
      output_data,
      indices_data,
      input_data,
      samples_data,
      numBatch,
      numPlane,
      inputSizeT,
      inputSizeH,
      inputSizeW,
      outputSizeT,
      outputSizeH,
      outputSizeW,
      poolSizeT,
      poolSizeH,
      poolSizeW,
      outputSize,
      iH_stride,
      iT_stride,
      iBatch_stride,
      oH_stride,
      oT_stride,
      oBatch_stride);

  sycl::range<3> local_range{(size_t)1, (size_t)1, (size_t)work_group_size};
  sycl::range<3> global_range{
      (size_t)numBatch,
      (size_t)outputSizeT,
      (size_t)(work_group_size * work_group_num)};
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

template <typename scalar_t>
struct FractionalMaxPool3dBackwardOutFrameKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    int ourOutputPoint = item.get_global_id()[2];
    int plane = item.get_group()[1];
    int batch = item.get_group()[0];

    if (ourOutputPoint < gradOutputPlaneSize_) {
      int64_t outputT = ourOutputPoint / gradOutputSizeH_ / gradOutputSizeW_;
      int64_t outputH = ourOutputPoint / gradOutputSizeW_ % gradOutputSizeH_;
      int64_t outputW = ourOutputPoint % gradOutputSizeW_;

      int64_t index = indices_[batch][plane][outputT][outputH][outputW];
      SYCL_KERNEL_ASSERT(index >= 0);
      int64_t inputW = index % gradInputSizeW_;
      int64_t inputH = (index / gradInputSizeW_ % gradInputSizeH_);
      int64_t inputT = index / (gradInputSizeH_ * gradInputSizeW_);
      SYCL_KERNEL_ASSERT(inputT < gradInput_.size(2));
      PackedTensorAccessor64<scalar_t, 5> gradInput_ptr = gradInput_;
      atomicAdd(
          (sycl_global_ptr<scalar_t>)&gradInput_ptr[batch][plane][inputT]
                                                   [inputH][inputW],
          gradOutput_[batch][plane][outputT][outputH][outputW]);
    }
  }
  FractionalMaxPool3dBackwardOutFrameKernelFunctor(
      PackedTensorAccessor64<scalar_t, 5> gradInput,
      PackedTensorAccessor64<scalar_t, 5> gradOutput,
      PackedTensorAccessor64<int64_t, 5> indices,
      int gradOutputSizeH,
      int gradOutputSizeW,
      int gradInputSizeH,
      int gradInputSizeW,
      int gradOutputPlaneSize)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        indices_(indices),
        gradOutputSizeH_(gradOutputSizeH),
        gradOutputSizeW_(gradOutputSizeW),
        gradInputSizeH_(gradInputSizeH),
        gradInputSizeW_(gradInputSizeW),
        gradOutputPlaneSize_(gradOutputPlaneSize) {}

 private:
  PackedTensorAccessor64<scalar_t, 5> gradInput_;
  PackedTensorAccessor64<scalar_t, 5> gradOutput_;
  PackedTensorAccessor64<int64_t, 5> indices_;
  int gradOutputSizeH_;
  int gradOutputSizeW_;
  int gradInputSizeH_;
  int gradInputSizeW_;
  int gradOutputPlaneSize_;
};

template <typename scalar_t>
void fractional_max_pool3d_backward_out_frame(
    PackedTensorAccessor64<scalar_t, 5> gradInput,
    PackedTensorAccessor64<scalar_t, 5> gradOutput,
    PackedTensorAccessor64<int64_t, 5> indices) {
  auto numBatch = gradInput.size(0);
  auto numPlane = gradInput.size(1);
  auto gradOutputSizeT = gradOutput.size(2);
  auto gradOutputSizeH = gradOutput.size(3);
  auto gradOutputSizeW = gradOutput.size(4);
  auto gradInputSizeH = gradInput.size(3);
  auto gradInputSizeW = gradInput.size(4);

  int gradOutputPlaneSize = gradOutputSizeT * gradOutputSizeH * gradOutputSizeW;
  int work_group_size = gradOutputPlaneSize > 256 ? 256 : gradOutputPlaneSize;
  int work_group_num =
      (gradOutputPlaneSize + work_group_size - 1) / work_group_size;

  FractionalMaxPool3dBackwardOutFrameKernelFunctor<scalar_t> kfn(
      gradInput,
      gradOutput,
      indices,
      gradOutputSizeH,
      gradOutputSizeW,
      gradInputSizeH,
      gradInputSizeW,
      gradOutputPlaneSize);
  sycl::range<3> local_range{(size_t)1, (size_t)1, (size_t)work_group_size};
  sycl::range<3> global_range{
      (size_t)numBatch,
      (size_t)numPlane,
      (size_t)(work_group_size * work_group_num)};
  sycl_kernel_submit(global_range, local_range, getCurrentSYCLQueue(), kfn);
}

void fractional_max_pool3d_out_kernel(
    const Tensor& input,
    int64_t poolSizeT,
    int64_t poolSizeH,
    int64_t poolSizeW,
    int64_t outputT,
    int64_t outputH,
    int64_t outputW,
    const Tensor& randomSamples,
    int64_t numBatch,
    int64_t numPlanes,
    int64_t inputT,
    int64_t inputH,
    int64_t inputW,
    const Tensor& output,
    const Tensor& indices) {
  int64_t planeDim = 0;
  int64_t dimt = 1;
  int64_t dimh = 2;
  int64_t dimw = 3;

  int64_t ndims = input.ndimension();

  if (ndims == 5) {
    numBatch = input.size(0);
    planeDim++;
    dimt++;
    dimh++;
    dimw++;
  }

  TORCH_CHECK(
      outputT + poolSizeT - 1 < inputT,
      "fractional_max_pool3d_out(): ",
      "pool time (",
      poolSizeT,
      ") too large relative to input time (",
      inputT,
      ")");
  TORCH_CHECK(
      outputH + poolSizeH - 1 < inputH,
      "fractional_max_pool3d_out(): ",
      "pool height (",
      poolSizeH,
      ") too large relative to input height (",
      inputH,
      ")");
  TORCH_CHECK(
      outputW + poolSizeW - 1 < inputW,
      "fractional_max_pool3d_out(): ",
      "pool width (",
      poolSizeW,
      ") too large relative to input width (",
      inputW,
      ")");

  auto smf = (4 == ndims) ? at::MemoryFormat::Contiguous
                          : input.suggest_memory_format();
  if (ndims == 4) {
    /* resize output */
    output.resize_({numPlanes, outputT, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputT, outputH, outputW});
  } else {
    /* resize output */
    output.resize_({numBatch, numPlanes, outputT, outputH, outputW}, smf);
    /* indices will contain the locations for each output point */
    indices.resize_({numBatch, numPlanes, outputT, outputH, outputW}, smf);
  }

  auto output_ = output;
  auto indices_ = indices;
  auto input_ = input.contiguous(smf);
  if (ndims == 4) {
    output_ = output_.reshape({1, numPlanes, outputT, outputH, outputW});
    indices_ = indices_.reshape({1, numPlanes, outputT, outputH, outputW});
    input_ = input_.reshape({1, numPlanes, inputT, inputH, inputW});
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fractional_max_pool3d_out_frame_xpu",
      [&] {
        if (is_smf_channels_last(input))
          fractional_max_pool3d_out_frame_cl<scalar_t>(
              output_.data_ptr<scalar_t>(),
              indices_.data_ptr<int64_t>(),
              input_.data_ptr<scalar_t>(),
              randomSamples.data_ptr<scalar_t>(),
              input_.size(0),
              input_.size(1),
              inputT,
              inputH,
              inputW,
              outputT,
              outputH,
              outputW,
              poolSizeT,
              poolSizeH,
              poolSizeW);
        else
          fractional_max_pool3d_out_frame_cf<scalar_t>(
              output_.data_ptr<scalar_t>(),
              indices_.data_ptr<int64_t>(),
              input_.data_ptr<scalar_t>(),
              randomSamples.data_ptr<scalar_t>(),
              input_.size(0),
              input_.size(1),
              inputT,
              inputH,
              inputW,
              outputT,
              outputH,
              outputW,
              poolSizeT,
              poolSizeH,
              poolSizeW);
      });
}

void fractional_max_pool3d_backward_out_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef pool_size /* unused */,
    IntArrayRef output_size,
    const Tensor& indices) {
  int64_t dimt = 1;
  int64_t dimh = 2;
  int64_t dimw = 3;

  int64_t outputT = output_size[0];
  int64_t outputH = output_size[1];
  int64_t outputW = output_size[2];

  int64_t ndims = input.ndimension();
  if (ndims == 5) {
    dimt++;
    dimh++;
    dimw++;
  }

  /* sizes */
  int64_t inputT = input.size(dimt);
  int64_t inputH = input.size(dimh);
  int64_t inputW = input.size(dimw);

  TORCH_CHECK(
      outputT == gradOutput.size(dimt),
      "fractional_max_pool3d_backward_out(): ",
      "gradOutput time unexpected");
  TORCH_CHECK(
      outputH == gradOutput.size(dimh),
      "fractional_max_pool3d_backward_out(): ",
      "gradOutput height unexpected");
  TORCH_CHECK(
      outputW == gradOutput.size(dimw),
      "fractional_max_pool3d_backward_out(): ",
      "gradOutput width unexpected");

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  auto gradInput_ = gradInput;
  auto gradOutput_ = gradOutput;
  auto indices_ = indices;

  if (ndims == 4) {
    gradInput_ =
        gradInput_.reshape({1, gradInput.size(0), inputT, inputH, inputW});
    gradOutput_ =
        gradOutput_.reshape({1, gradOutput.size(0), outputT, outputH, outputW});
    indices_ =
        indices_.reshape({1, indices.size(0), outputT, outputH, outputW});
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gradOutput.scalar_type(),
      "fractional_max_pool3d_backward_xpu",
      [&] {
        fractional_max_pool3d_backward_out_frame<scalar_t>(
            gradInput_.packed_accessor64<scalar_t, 5>(),
            gradOutput_.packed_accessor64<scalar_t, 5>(),
            indices_.packed_accessor64<int64_t, 5>());
      });
}

} // namespace at::native::xpu