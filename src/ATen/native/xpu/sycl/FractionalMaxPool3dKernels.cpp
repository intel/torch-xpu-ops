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
#include <ATen/native/FractionalMaxPooling.h>
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

template <typename scalar_t>
struct FractionalMaxPool3dOutFrameFunctor {
  void operator()(sycl::nd_item<3> item) const {
    using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
    // Output (t, h, w) point that this thread is responsible for
    int64_t ourOutputPoint =
        item.get_local_id(2) + item.get_group(2) * item.get_local_range(2);
    int64_t plane = item.get_group(1);
    int64_t batch = item.get_group(0);
    // Each thread generates a specific output point
    if (ourOutputPoint < output_.size(2) * output_.size(3) * output_.size(4)) {
      int64_t outputT = ourOutputPoint / (output_.size(3) * output_.size(4));
      int64_t outputH = (ourOutputPoint / output_.size(4)) % output_.size(3);
      int64_t outputW = ourOutputPoint % output_.size(4);

      int64_t poolT = get_intervals<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(samples_[batch][plane][0]),
          outputT,
          input_.size(2),
          output_.size(2),
          poolSizeT_);
      int64_t poolH = get_intervals<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(samples_[batch][plane][1]),
          outputH,
          input_.size(3),
          output_.size(3),
          poolSizeH_);
      int64_t poolW = get_intervals<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(samples_[batch][plane][2]),
          outputW,
          input_.size(4),
          output_.size(4),
          poolSizeW_);

      scalar_t maxVal = at::numeric_limits<scalar_t>::lower_bound();
      int64_t maxIndex = poolT * input_.size(3) * input_.size(4) +
          poolH * input_.size(4) + poolW;

      for (int64_t t = poolT; t < poolT + poolSizeT_; ++t) {
        for (int64_t h = poolH; h < poolH + poolSizeH_; ++h) {
          if (poolSizeW_ < 2 || poolSizeW_ > 7) {
            for (int64_t w = poolW; w < poolW + poolSizeW_; ++w) {
              scalar_t val = input_[batch][plane][t][h][w];
              // for consistency with THNN, favor the first max
              if (val > maxVal || at::_isnan(val)) {
                maxIndex = t * input_.size(3) * input_.size(4) +
                    h * input_.size(4) + w;
                maxVal = val;
              }
            }
          } else {
            for (int64_t i = 0; i < poolSizeW_; ++i) {
              int64_t w = i + poolW;
              scalar_t val = input_[batch][plane][t][h][w];
              // for consistency with THNN, favor the first max
              if (val > maxVal || at::_isnan(val)) {
                maxIndex = t * input_.size(3) * input_.size(4) +
                    h * input_.size(4) + w;
                maxVal = val;
              }
            }
          }
        }
      }
      auto indices_acc = indices_;
      auto output_acc = output_;
      indices_acc[batch][plane][outputT][outputH][outputW] = maxIndex;
      output_acc[batch][plane][outputT][outputH][outputW] = maxVal;
    }
  }
  FractionalMaxPool3dOutFrameFunctor(
      PackedTensorAccessor64<const scalar_t, 5> input,
      PackedTensorAccessor64<scalar_t, 5> output,
      PackedTensorAccessor64<int64_t, 5> indices,
      PackedTensorAccessor64<const scalar_t, 3> samples,
      int64_t poolSizeT,
      int64_t poolSizeH,
      int64_t poolSizeW)
      : input_(input),
        output_(output),
        indices_(indices),
        samples_(samples),
        poolSizeT_(poolSizeT),
        poolSizeH_(poolSizeH),
        poolSizeW_(poolSizeW) {}

 private:
  PackedTensorAccessor64<const scalar_t, 5> input_;
  PackedTensorAccessor64<scalar_t, 5> output_;
  PackedTensorAccessor64<int64_t, 5> indices_;
  PackedTensorAccessor64<const scalar_t, 3> samples_;
  int64_t poolSizeT_;
  int64_t poolSizeH_;
  int64_t poolSizeW_;
};

void fractional_max_pool3d_kernel(
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
  fractional_max_pool_check_shape</*ndim*/ 3>(input, randomSamples);

  auto output_ = output;
  auto indices_ = indices;
  auto input_ = input;

  int64_t ndims = input_.ndimension();
  if (ndims == 4) {
    output_ = output_.reshape({1, numPlanes, outputT, outputH, outputW});
    indices_ = indices_.reshape({1, numPlanes, outputT, outputH, outputW});
    input_ = input_.reshape({1, numPlanes, inputT, inputH, inputW});
  }
  if (output_.numel() == 0) {
    return;
  }

  int64_t outputPlaneSize = output_.size(2) * output_.size(3) * output_.size(4);
  size_t group_x = outputPlaneSize > 128 ? 128 : outputPlaneSize;
  size_t nwg_x = (outputPlaneSize + 127) / 128;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fractional_max_pool3d_out_frame_xpu",
      [&] {
        auto kfn = FractionalMaxPool3dOutFrameFunctor<scalar_t>(
            input_.packed_accessor64<const scalar_t, 5>(),
            output_.packed_accessor64<scalar_t, 5>(),
            indices_.packed_accessor64<int64_t, 5>(),
            randomSamples.packed_accessor64<const scalar_t, 3>(),
            poolSizeT,
            poolSizeH,
            poolSizeW);
        sycl::range<3> local_range(1, 1, group_x);
        sycl::range<3> global_range(
            input_.size(0), input_.size(1), group_x * nwg_x);
        sycl_kernel_submit(
            global_range, local_range, getCurrentSYCLQueue(), kfn);
      });
}

template <typename scalar_t>
struct FractionalMaxPool3dBackwardOutFrameKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    // Output (h, w) point that this thread is responsible for
    int64_t ourOutputPoint =
        item.get_local_id(2) + item.get_group(2) * item.get_local_range(2);
    int64_t plane = item.get_group(1);
    int64_t batch = item.get_group(0);

    // Each thread generates a specific output point
    if (ourOutputPoint <
        gradOutput_.size(2) * gradOutput_.size(3) * gradOutput_.size(4)) {
      int64_t outputW = ourOutputPoint % gradOutput_.size(4);
      int64_t outputH =
          (ourOutputPoint / gradOutput_.size(4)) % gradOutput_.size(3);
      int64_t outputT =
          ourOutputPoint / (gradOutput_.size(3) * gradOutput_.size(4));

      int64_t index = indices_[batch][plane][outputT][outputH][outputW];
      SYCL_KERNEL_ASSERT(index >= 0);
      int64_t inputW = index % gradInput_.size(4);
      int64_t inputH = (index / gradInput_.size(4)) % gradInput_.size(3);
      int64_t inputT = index / (gradInput_.size(3) * gradInput_.size(4));
      SYCL_KERNEL_ASSERT(inputT < gradInput_.size(2));

      auto gradInput_acc = gradInput_;

      atomicAdd(
          (sycl_global_ptr<scalar_t>)&gradInput_acc[batch][plane][inputT]
                                                   [inputH][inputW],
          gradOutput_[batch][plane][outputT][outputH][outputW]);
    }
  }
  FractionalMaxPool3dBackwardOutFrameKernelFunctor(
      PackedTensorAccessor64<scalar_t, 5> gradInput,
      PackedTensorAccessor64<const scalar_t, 5> gradOutput,
      PackedTensorAccessor64<const int64_t, 5> indices)
      : gradInput_(gradInput), gradOutput_(gradOutput), indices_(indices) {}

 private:
  PackedTensorAccessor64<scalar_t, 5> gradInput_;
  PackedTensorAccessor64<const scalar_t, 5> gradOutput_;
  PackedTensorAccessor64<const int64_t, 5> indices_;
};

void fractional_max_pool3d_backward_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
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

  if (gradInput.numel() == 0) {
    return;
  }

  int64_t outputPlaneSize =
      gradOutput_.size(2) * gradOutput_.size(3) * gradOutput_.size(4);
  size_t group_x = outputPlaneSize > 128 ? 128 : outputPlaneSize;
  size_t nwg_x = (outputPlaneSize + 127) / 128;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gradOutput.scalar_type(),
      "fractional_max_pool3d_backward_xpu",
      [&] {
        auto kfn = FractionalMaxPool3dBackwardOutFrameKernelFunctor<scalar_t>(
            gradInput_.packed_accessor64<scalar_t, 5>(),
            gradOutput_.packed_accessor64<const scalar_t, 5>(),
            indices_.packed_accessor64<const int64_t, 5>());
        sycl::range<3> local_range(1, 1, group_x);
        sycl::range<3> global_range(
            gradInput_.size(0), gradInput_.size(1), group_x * nwg_x);
        sycl_kernel_submit(
            global_range, local_range, getCurrentSYCLQueue(), kfn);
      });
}

} // namespace at::native::xpu

#pragma clang diagnostic pop
#pragma GCC diagnostic pop
