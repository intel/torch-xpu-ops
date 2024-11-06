#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/FractionalMaxPooling.h>
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

template <typename scalar_t>
struct FractionalMaxPool2dOutFrameKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    using accscalar_t = at::acc_type_device<scalar_t, kXPU>;

    int ourOutputPoint =
        item.get_local_id(2) + item.get_group(2) * item.get_local_range(2);
    int plane = item.get_group(1);
    int batch = item.get_group(0);

    // Each thread generates a specific output point
    if (ourOutputPoint < output_.size(2) * output_.size(3)) {
      int outputW = ourOutputPoint % output_.size(3);
      int outputH = ourOutputPoint / output_.size(3);

      int poolW = get_interval<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(samples_[batch][plane][0]),
          outputW,
          input_.size(3),
          output_.size(3),
          poolSizeW_);
      int poolH = get_interval<scalar_t, accscalar_t>(
          static_cast<accscalar_t>(samples_[batch][plane][1]),
          outputH,
          input_.size(2),
          output_.size(2),
          poolSizeH_);

      scalar_t maxVal = at::numeric_limits<scalar_t>::lower_bound();
      int maxIndex = poolH * input_.size(3) + poolW;

      for (int h = poolH; h < poolH + poolSizeH_; ++h) {
        if (poolSizeW_ < 2 || poolSizeW_ > 7) {
          for (int w = poolW; w < poolW + poolSizeW_; ++w) {
            scalar_t val = input_[batch][plane][h][w];
            // for consistency with THNN, favor the first max
            if (val > maxVal || at::_isnan(val)) {
              maxIndex = h * input_.size(3) + w;
              maxVal = val;
            }
          }
        } else {
          for (int i = 0; i < poolSizeW_; ++i) {
            int w = i + poolW;
            scalar_t val = input_[batch][plane][h][w];
            // for consistency with THNN, favor the first max
            if (val > maxVal || at::_isnan(val)) {
              maxIndex = h * input_.size(3) + w;
              maxVal = val;
            }
          }
        }
      }

      auto indices_acc = indices_;
      auto output_acc = output_;

      indices_acc[batch][plane][outputH][outputW] = maxIndex;
      output_acc[batch][plane][outputH][outputW] = maxVal;
    }
  }
  FractionalMaxPool2dOutFrameKernelFunctor(
      PackedTensorAccessor<scalar_t, 4> output,
      PackedTensorAccessor<int64_t, 4> indices,
      PackedTensorAccessor<const scalar_t, 4> input,
      PackedTensorAccessor<const scalar_t, 3> samples,
      int poolSizeH,
      int poolSizeW)
      : output_(output),
        indices_(indices),
        input_(input),
        samples_(samples),
        poolSizeH_(poolSizeH),
        poolSizeW_(poolSizeW) {}

 private:
  PackedTensorAccessor<scalar_t, 4> output_;
  PackedTensorAccessor<int64_t, 4> indices_;
  PackedTensorAccessor<const scalar_t, 4> input_;
  PackedTensorAccessor<const scalar_t, 3> samples_;
  int poolSizeH_;
  int poolSizeW_;
};

void fractional_max_pool2d_kernel(
    const Tensor& input,
    IntArrayRef pool_size,
    IntArrayRef output_size,
    const Tensor& randomSamples,
    const Tensor& output,
    const Tensor& indices) {
  fractional_max_pool_check_shape</*ndim*/ 2>(input, randomSamples);

  int planeDim = 0;
  int ndims = input.ndimension();
  if (ndims == 4) {
    planeDim++;
  }

  /* sizes */
  int numPlanes = input.size(planeDim);

  int outputH = output_size[0];
  int outputW = output_size[1];
  int poolSizeH = pool_size[0];
  int poolSizeW = pool_size[1];

  auto output_ = output;
  auto input_ = input;
  auto indices_ = indices;

  if (ndims == 3) {
    output_ = output_.reshape({1, numPlanes, outputH, outputW});
    indices_ = indices_.reshape({1, numPlanes, outputH, outputW});
    input_ = input_.reshape({1, input.size(0), input.size(1), input.size(2)});
  }

  if (output_.numel() == 0) {
    return;
  }

  int outputPlaneSize = output_.size(2) * output_.size(3);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      input.scalar_type(),
      "fractional_max_pool2d_out_xpu_frame",
      [&] {
        auto devInput = input_.packed_accessor64<const scalar_t, 4>();
        auto devOutput = output_.packed_accessor64<scalar_t, 4>();
        auto devIndices = indices_.packed_accessor64<int64_t, 4>();
        auto devSamples = randomSamples.packed_accessor64<const scalar_t, 3>();
        auto kfn = FractionalMaxPool2dOutFrameKernelFunctor<scalar_t>(
            devOutput, devIndices, devInput, devSamples, poolSizeH, poolSizeW);
        size_t group_x = outputPlaneSize > 128 ? 128 : outputPlaneSize;
        size_t nwg_x = (outputPlaneSize + 127) / 128;
        sycl::range<3> local_range(1, 1, group_x);
        sycl::range<3> global_range(
            input_.size(0), input_.size(1), nwg_x * group_x);
        sycl_kernel_submit(
            global_range, local_range, getCurrentSYCLQueue(), kfn);
      });
}

template <typename scalar_t>
struct FractionalMaxPool2dBackwardOutFrameKernelFunctor {
  void operator()(sycl::nd_item<3> item) const {
    // Output (h, w) point that this thread is responsible for
    int ourOutputPoint =
        item.get_local_id(2) + item.get_group(2) * item.get_local_range(2);
    int plane = item.get_group(1);
    int batch = item.get_group(0);

    // Each thread generates a specific output point
    if (ourOutputPoint < gradOutput_.size(2) * gradOutput_.size(3)) {
      int outputW = ourOutputPoint % gradOutput_.size(3);
      int outputH = ourOutputPoint / gradOutput_.size(3);

      int index = indices_[batch][plane][outputH][outputW];
      SYCL_KERNEL_ASSERT(index >= 0);
      int inputW = index % gradInput_.size(3);
      int inputH = index / gradInput_.size(3);
      SYCL_KERNEL_ASSERT(inputH < gradInput_.size(2));

      auto gradInput_acc = gradInput_;

      atomicAdd(
          (sycl_global_ptr<scalar_t>)&gradInput_acc[batch][plane][inputH]
                                                   [inputW],
          gradOutput_[batch][plane][outputH][outputW]);
    }
  }
  FractionalMaxPool2dBackwardOutFrameKernelFunctor(
      PackedTensorAccessor<scalar_t, 4> gradInput,
      PackedTensorAccessor<const scalar_t, 4> gradOutput,
      PackedTensorAccessor<const int64_t, 4> indices)
      : gradInput_(gradInput), gradOutput_(gradOutput), indices_(indices) {}

 private:
  PackedTensorAccessor<scalar_t, 4> gradInput_;
  PackedTensorAccessor<const scalar_t, 4> gradOutput_;
  PackedTensorAccessor<const int64_t, 4> indices_;
};

void fractional_max_pool2d_backward_kernel(
    const Tensor& gradOutput,
    const Tensor& input,
    IntArrayRef pool_size /* unused */,
    IntArrayRef output_size,
    const Tensor& indices,
    const Tensor& gradInput) {
  globalContext().alertNotDeterministic("fractional_max_pool2d_backward_xpu");

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

  if (gradInput.numel() == 0) {
    return;
  }

  gradInput.zero_();

  auto gradInput_ = gradInput;
  auto gradOutput_ = gradOutput;
  auto indices_ = indices;

  if (ndims == 3) {
    gradInput_ = gradInput_.reshape({1, input.size(0), inputH, inputW});
    gradOutput_ =
        gradOutput_.reshape({1, gradOutput.size(0), outputH, outputW});
    indices_ = indices_.reshape({1, indices_.size(0), outputH, outputW});
  }

  int outputPlaneSize = gradOutput_.size(2) * gradOutput_.size(3);
  auto devIndices = indices_.packed_accessor64<const int64_t, 4>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gradOutput.scalar_type(),
      "fractional_max_pool2d_backward_out_xpu_frame",
      [&] {
        auto devGradInput = gradInput_.packed_accessor64<scalar_t, 4>();
        auto devGradOutput = gradOutput_.packed_accessor64<const scalar_t, 4>();
        auto kfn = FractionalMaxPool2dBackwardOutFrameKernelFunctor<scalar_t>(
            devGradInput, devGradOutput, devIndices);
        size_t group_x = outputPlaneSize > 128 ? 128 : outputPlaneSize;
        size_t nwg_x = (outputPlaneSize + 127) / 128;
        sycl::range<3> local_range(1, 1, group_x);
        sycl::range<3> global_range(
            gradInput_.size(0), gradInput_.size(1), nwg_x * group_x);
        sycl_kernel_submit(
            global_range, local_range, getCurrentSYCLQueue(), kfn);
      });
}

} // namespace at::native::xpu

#pragma clang diagnostic pop
#pragma GCC diagnostic pop
