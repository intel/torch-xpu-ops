#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/MemoryFormat.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/FractionalMaxPool2dKernels.h>

namespace at::native::xpu {

template <typename scalar_t, typename accscalar_t>
inline int get_interval(
    accscalar_t sample,
    int index,
    int inputSize,
    int outputSize,
    int poolSize) {
  accscalar_t alpha = static_cast<accscalar_t>(inputSize - poolSize) /
      static_cast<accscalar_t>(outputSize - 1);
  if (index == outputSize - 1) {
    return inputSize - poolSize;
  } else {
    return static_cast<int>((index + sample) * alpha) -
        static_cast<int>(sample * alpha);
  }
}

template <typename scalar_t, typename accscalar_t>
struct FractionalMaxPool2dOutFrameKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto input_ptr = input_data_;
    auto output_ptr = output_data_;
    auto indices_ptr = indices_data_;
    auto samples_ptr = samples_data_;

    int linearIndex = item.get_global_id()[0];
    for (int l = 0; l < loops_; ++l) {
      int outputIndex = linearIndex + l * (work_group_size_ * work_group_num_);
      int batch = outputIndex / (numPlane_ * outputSizeH_ * outputSizeW_);
      int plane = is_channels_last_
          ? outputIndex % numPlane_
          : (outputIndex / outputSizeH_ / outputSizeW_) % numPlane_;
      int outputH = is_channels_last_
          ? outputIndex / numPlane_ / outputSizeW_ % outputSizeH_
          : outputIndex / outputSizeW_ % outputSizeH_;
      int outputW = is_channels_last_ ? outputIndex / numPlane_ % outputSizeW_
                                      : outputIndex % outputSizeW_;

      if (batch < numBatch_ && plane < numPlane_ && outputH < outputSizeH_ &&
          outputW < outputSizeW_) {
        int poolW = get_interval<scalar_t, accscalar_t>(
            static_cast<accscalar_t>(samples_ptr
                                         [batch * numPlane_ * 2 + plane * 2 +
                                          0] /*[batch][plane][0] */),
            outputW,
            inputSizeW_,
            outputSizeW_,
            poolSizeW_);
        int poolH = get_interval<scalar_t, accscalar_t>(
            static_cast<accscalar_t>(samples_ptr
                                         [batch * numPlane_ * 2 + plane * 2 +
                                          1] /*[batch][plane][1] */),
            outputH,
            inputSizeH_,
            outputSizeH_,
            poolSizeH_);

        scalar_t maxVal = std::numeric_limits<scalar_t>::lowest();
        int maxIndex = -1;

        for (int h = poolH; h < poolH + poolSizeH_; ++h) {
          for (int w = poolW; w < poolW + poolSizeW_; ++w) {
            int64_t load_offset = is_channels_last_
                ? batch * inputSizeH_ * inputSizeW_ * numPlane_ + plane +
                    h * inputSizeW_ * numPlane_ + w * numPlane_
                : batch * numPlane_ * inputSizeH_ * inputSizeW_ +
                    plane * inputSizeH_ * inputSizeW_ + h * inputSizeW_ + w;
            scalar_t val = input_ptr[load_offset];
            if (val > maxVal) {
              maxIndex = h * inputSizeW_ + w;
              maxVal = val;
            }
          }
        }

        int64_t store_offset = is_channels_last_
            ? batch * outputSizeH_ * outputSizeW_ * numPlane_ + plane +
                outputH * outputSizeW_ * numPlane_ + outputW * numPlane_
            : batch * numPlane_ * outputSizeH_ * outputSizeW_ +
                plane * outputSizeH_ * outputSizeW_ + outputH * outputSizeW_ +
                outputW;
        indices_ptr[store_offset] = maxIndex;
        output_ptr[store_offset] = maxVal;
      }
    }
  }
  FractionalMaxPool2dOutFrameKernelFunctor(
      scalar_t* output_data,
      int64_t* indices_data,
      scalar_t* input_data,
      scalar_t* samples_data,
      int numBatch,
      int numPlane,
      int inputSizeH,
      int inputSizeW,
      int outputSizeH,
      int outputSizeW,
      int poolSizeH,
      int poolSizeW,
      const bool is_channels_last,
      int work_group_size,
      int work_group_num,
      int loops)
      : output_data_(output_data),
        indices_data_(indices_data),
        input_data_(input_data),
        samples_data_(samples_data),
        numBatch_(numBatch),
        numPlane_(numPlane),
        inputSizeH_(inputSizeH),
        inputSizeW_(inputSizeW),
        outputSizeH_(outputSizeH),
        outputSizeW_(outputSizeW),
        poolSizeH_(poolSizeH),
        poolSizeW_(poolSizeW),
        is_channels_last_(is_channels_last),
        work_group_size_(work_group_size),
        work_group_num_(work_group_num),
        loops_(loops) {}

 private:
  scalar_t* output_data_;
  int64_t* indices_data_;
  scalar_t* input_data_;
  scalar_t* samples_data_;
  int numBatch_;
  int numPlane_;
  int inputSizeH_;
  int inputSizeW_;
  int outputSizeH_;
  int outputSizeW_;
  int poolSizeH_;
  int poolSizeW_;
  const bool is_channels_last_;
  int work_group_size_;
  int work_group_num_;
  int loops_;
};

template <typename scalar_t>
void fractional_max_pool2d_out_xpu_frame(
    scalar_t* output,
    int64_t* indices,
    scalar_t* input,
    scalar_t* samples,
    int numBatch,
    int numPlane,
    int inputSizeH,
    int inputSizeW,
    int outputSizeH,
    int outputSizeW,
    int poolSizeH,
    int poolSizeW,
    const bool is_channels_last) {
  using accscalar_t = acc_type_device<scalar_t, kXPU>;
  using KernelClass =
      FractionalMaxPool2dOutFrameKernelFunctor<scalar_t, accscalar_t>;

  int64_t max_wg_size = syclMaxWorkGroupSize<KernelClass>();
  int outputSize = numBatch * numPlane * outputSizeH * outputSizeW;
  int work_group_size = outputSize > max_wg_size ? max_wg_size : outputSize;
  // One full device launch could launch en_num * SMID32 * HD threads as below
  const auto target_global_size = syclMaxWorkItemsPerTile();
  // Each work group size is work_group_size, one full device launch is
  // target_global_size, so we can calculate max work group num as below
  const int max_work_group_num = target_global_size / work_group_size;
  int work_group_num = outputSize / work_group_size < max_work_group_num
      ? outputSize / work_group_size
      : max_work_group_num;
  int draft_work_group_num =
      (outputSize + work_group_size - 1) / work_group_size;
  // work item in each work group calculates loops' elements
  int loops = draft_work_group_num / work_group_num + 1;

  auto kfn = KernelClass(
      output,
      indices,
      input,
      samples,
      numBatch,
      numPlane,
      inputSizeH,
      inputSizeW,
      outputSizeH,
      outputSizeW,
      poolSizeH,
      poolSizeW,
      is_channels_last,
      work_group_size,
      work_group_num,
      loops);

  sycl_kernel_submit(
      work_group_size * work_group_num,
      work_group_size,
      getCurrentSYCLQueue(),
      kfn);
}

template <typename scalar_t, bool is_channels_last>
struct FractionalMaxPool2dBackwardOutFrameKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto gradInput_ptr = gradInput_data_;
    auto gradOutput_ptr = gradOutput_data_;
    auto indices_ptr = indices_data_;

    int64_t outputIndex = item.get_global_id()[0];
    if (outputIndex < gradOutputSize_) {
      int batch = outputIndex / out_n_stride_;
      if constexpr (is_channels_last) {
        int plane = outputIndex % numPlane_;
        int64_t index = indices_ptr[outputIndex];
        int64_t gI_offset = batch * in_n_stride_ + plane + index * numPlane_;
        atomicAdd(
            (sycl_global_ptr<scalar_t>)&gradInput_ptr[gI_offset],
            gradOutput_ptr[outputIndex]);
      } else {
        int plane = outputIndex / out_cf_c_stride_ % numPlane_;
        int64_t index = indices_ptr[outputIndex];
        int64_t gI_offset =
            batch * in_n_stride_ + plane * in_cf_c_stride_ + index;
        atomicAdd(
            (sycl_global_ptr<scalar_t>)&gradInput_ptr[gI_offset],
            gradOutput_ptr[outputIndex]);
      }
    }
  }
  FractionalMaxPool2dBackwardOutFrameKernelFunctor(
      scalar_t* gradInput_data,
      scalar_t* gradOutput_data,
      int64_t* indices_data,
      int numBatch,
      int numPlane,
      int gradInputSizeW,
      int64_t gradOutputSize,
      int out_cf_c_stride,
      int in_cf_c_stride,
      int out_n_stride,
      int in_n_stride)
      : gradInput_data_(gradInput_data),
        gradOutput_data_(gradOutput_data),
        indices_data_(indices_data),
        numBatch_(numBatch),
        numPlane_(numPlane),
        gradInputSizeW_(gradInputSizeW),
        gradOutputSize_(gradOutputSize),
        out_cf_c_stride_(out_cf_c_stride),
        in_cf_c_stride_(in_cf_c_stride),
        out_n_stride_(out_n_stride),
        in_n_stride_(in_n_stride) {}

 private:
  scalar_t* gradInput_data_;
  scalar_t* gradOutput_data_;
  int64_t* indices_data_;
  int numBatch_;
  int numPlane_;
  int gradInputSizeW_;
  int64_t gradOutputSize_;
  int out_cf_c_stride_;
  int in_cf_c_stride_;
  int out_n_stride_;
  int in_n_stride_;
};

template <typename scalar_t, bool is_channels_last>
void fractional_max_pool2d_backward_out_xpu_frame(
    scalar_t* gradInput,
    scalar_t* gradOutput,
    int64_t* indices,
    int numBatch,
    int numPlane,
    int gradInputSizeH,
    int gradInputSizeW,
    int gradOutputSizeH,
    int gradOutputSizeW) {
  using KernelClass = FractionalMaxPool2dBackwardOutFrameKernelFunctor<
      scalar_t,
      is_channels_last>;

  int64_t max_wg_size = syclMaxWorkGroupSize<KernelClass>();
  int64_t gradOutputSize =
      numBatch * numPlane * gradOutputSizeH * gradOutputSizeW;
  int work_group_size =
      gradOutputSize > max_wg_size ? max_wg_size : gradOutputSize;
  int global_range =
      ((gradOutputSize - 1) / work_group_size + 1) * work_group_size;
  auto out_cf_c_stride = gradOutputSizeH * gradOutputSizeW;
  auto in_cf_c_stride = gradInputSizeH * gradInputSizeW;
  auto out_n_stride = numPlane * out_cf_c_stride;
  auto in_n_stride = numPlane * in_cf_c_stride;

  auto kfn = KernelClass(
      gradInput,
      gradOutput,
      indices,
      numBatch,
      numPlane,
      gradInputSizeW,
      gradOutputSize,
      out_cf_c_stride,
      in_cf_c_stride,
      out_n_stride,
      in_n_stride);
  sycl_kernel_submit(global_range, work_group_size, getCurrentSYCLQueue(), kfn);
}

void fractional_max_pool2d_out_kernel(
    const Tensor& output,
    const Tensor& indices,
    const Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const Tensor& randomSamples) {
  int planeDim = 0;
  int dimh = 1;
  int dimw = 2;
  int numBatch = 1;

  int ndims = input.ndimension();
  TORCH_CHECK(
      input.numel() > 0,
      "fractional_max_pool2d(): expected input to have non-empty ",
      "spatial dimensions.");

  TORCH_CHECK(
      (ndims == 3 || ndims == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  if (ndims == 4) {
    numBatch = input.size(0);
    planeDim++;
    dimh++;
    dimw++;
  }

  /* sizes */
  int numPlanes = input.size(planeDim);
  int inputH = input.size(dimh);
  int inputW = input.size(dimw);

  int outputH = output_size[0];
  int outputW = output_size[1];
  int poolSizeH = pool_size[0];
  int poolSizeW = pool_size[1];

  TORCH_CHECK(
      outputH + poolSizeH - 1 <= inputH,
      "fractional_max_pool2d(): pool_size height ",
      poolSizeH,
      " too large relative to input height ",
      inputH);
  TORCH_CHECK(
      outputW + poolSizeW - 1 <= inputW,
      "pool_size width ",
      poolSizeW,
      " too large relative to input width ",
      inputW);

  auto smf = (3 == ndims) ? at::MemoryFormat::Contiguous
                          : input.suggest_memory_format();

  if (ndims == 3) {
    /* resize output */
    output.resize_({numPlanes, outputH, outputW});
    /* indices will contain the locations for each output point */
    indices.resize_({numPlanes, outputH, outputW});
  } else {
    output.resize_({numBatch, numPlanes, outputH, outputW}, smf);
    indices.resize_({numBatch, numPlanes, outputH, outputW}, smf);
  }

  auto output_ = output;
  auto input_ = input.contiguous(smf);
  auto indices_ = indices;

  if (ndims == 3) {
    output_ = output_.reshape({1, numPlanes, outputH, outputW});
    indices_ = indices_.reshape({1, numPlanes, outputH, outputW});
    input_ = input_.reshape({1, input.size(0), input.size(1), input.size(2)});
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input.scalar_type(),
      "fractional_max_pool2d_out_xpu_frame",
      [&] {
        fractional_max_pool2d_out_xpu_frame<scalar_t>(
            output_.data_ptr<scalar_t>(),
            indices_.data_ptr<int64_t>(),
            input_.data_ptr<scalar_t>(),
            randomSamples.data_ptr<scalar_t>(),
            input_.size(0),
            input_.size(1),
            inputH,
            inputW,
            outputH,
            outputW,
            poolSizeH,
            poolSizeW,
            is_smf_channels_last(input));
      });
}

void fractional_max_pool2d_backward_kernel(
    const Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef pool_size /* unused */,
    IntArrayRef output_size,
    const Tensor& indices) {
  int dimh = 1;
  int dimw = 2;

  int ndims = input.ndimension();
  if (ndims == 4) {
    dimh++;
    dimw++;
  }

  /* sizes */
  int inputH = input.size(dimh);
  int inputW = input.size(dimw);

  int outputH = output_size[0];
  int outputW = output_size[1];

  TORCH_CHECK(
      outputH == gradOutput.size(dimh),
      "fractional_max_pool2d(): gradOutput height unexpected");
  TORCH_CHECK(
      outputW == gradOutput.size(dimw),
      "fractional_max_pool2d(): gradOutput width unexpected");

  auto smf = (3 == ndims) ? at::MemoryFormat::Contiguous
                          : input.suggest_memory_format();

  /* resize */
  gradInput.resize_as_(input, smf);
  gradInput.zero_();

  auto gradInput_ = gradInput;
  auto gradOutput_ = gradOutput.contiguous(smf);
  auto indices_ = indices;
  if (ndims == 3) {
    gradInput_ = gradInput_.reshape({1, input.size(0), inputH, inputW});
    gradOutput_ =
        gradOutput_.reshape({1, gradOutput.size(0), outputH, outputW});
    indices_ = indices_.reshape({1, indices_.size(0), outputH, outputW});
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gradOutput.scalar_type(),
      "fractional_max_pool2d_backward_out_xpu_frame",
      [&] {
        if (is_smf_channels_last(input)) {
          fractional_max_pool2d_backward_out_xpu_frame<scalar_t, true>(
              gradInput_.data_ptr<scalar_t>(),
              gradOutput_.data_ptr<scalar_t>(),
              indices_.data_ptr<int64_t>(),
              gradInput_.size(0),
              gradInput_.size(1),
              inputH,
              inputW,
              outputH,
              outputW);
        } else {
          fractional_max_pool2d_backward_out_xpu_frame<scalar_t, false>(
              gradInput_.data_ptr<scalar_t>(),
              gradOutput_.data_ptr<scalar_t>(),
              indices_.data_ptr<int64_t>(),
              gradInput_.size(0),
              gradInput_.size(1),
              inputH,
              inputW,
              outputH,
              outputW);
        }
      });
}

} // namespace at::native::xpu
