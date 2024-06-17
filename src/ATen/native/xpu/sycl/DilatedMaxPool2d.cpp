#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

namespace at {
namespace native {
namespace xpu {

static inline int p_start(
    int size,
    int pad,
    int kernel,
    int dilation,
    int stride) {
  return (size + pad < ((kernel - 1) * dilation + 1))
      ? 0
      : (size + pad - ((kernel - 1) * dilation + 1)) / stride + 1;
}

static inline int p_end(int size, int pad, int pooled_size, int stride) {
  return std::min((size + pad) / stride + 1, pooled_size);
}

template <typename scalar_t, bool is_channels_last>
struct MaxPool2dBackwardOutKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);

    do {
      if (desc.glb_problem < cfg_.problem_) {
        int batch = desc.glb_problem / out_n_stride_;
        int outputIndex = desc.glb_problem;
        if constexpr (is_channels_last) {
          int plane = outputIndex % numPlane_;
          int64_t index = indices_[outputIndex];
          int64_t gI_offset = batch * in_n_stride_ + plane + index * numPlane_;
          atomicAdd(
              (sycl_global_ptr<scalar_t>)&gradInput_[gI_offset],
              gradOutput_[outputIndex]);
        } else {
          int plane = outputIndex / out_cf_c_stride_ % numPlane_;
          int64_t index = indices_[outputIndex];
          int64_t gI_offset =
              batch * in_n_stride_ + plane * in_cf_c_stride_ + index;
          atomicAdd(
              (sycl_global_ptr<scalar_t>)&gradInput_[gI_offset],
              gradOutput_[outputIndex]);
        }
      }
    } while (cfg_.next(item, desc));
  }
  MaxPool2dBackwardOutKernelFunctor(
      scalar_t* gradInput,
      scalar_t* gradOutput,
      int64_t* indices,
      int numPlane,
      int gradInputSizeH,
      int gradInputSizeW,
      int gradOutputSizeH,
      int gradOutputSizeW,
      int64_t gradOutputSize,
      int out_cf_c_stride,
      int in_cf_c_stride,
      int out_n_stride,
      int in_n_stride,
      BatchKernelConfig cfg)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        indices_(indices),
        numPlane_(numPlane),
        gradInputSizeH_(gradInputSizeH),
        gradInputSizeW_(gradInputSizeW),
        gradOutputSizeH_(gradOutputSizeH),
        gradOutputSizeW_(gradOutputSizeW),
        gradOutputSize_(gradOutputSize),
        out_cf_c_stride_(out_cf_c_stride),
        in_cf_c_stride_(in_cf_c_stride),
        out_n_stride_(out_n_stride),
        in_n_stride_(in_n_stride),
        cfg_(cfg) {}

 private:
  scalar_t* gradInput_;
  scalar_t* gradOutput_;
  int64_t* indices_;
  int numPlane_;
  int gradInputSizeH_;
  int gradInputSizeW_;
  int gradOutputSizeH_;
  int gradOutputSizeW_;
  int64_t gradOutputSize_;
  int out_cf_c_stride_;
  int in_cf_c_stride_;
  int out_n_stride_;
  int in_n_stride_;
  BatchKernelConfig cfg_;
};

template <typename scalar_t, bool is_channels_last>
struct MaxPool2dBackwardOutDeterministicKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);
    do {
      if (desc.glb_problem < cfg_.problem_) {
        int inputIndex = desc.glb_problem;
        int batch = inputIndex / in_n_stride_;
        int plane;
        int64_t input_hw_index;
        if constexpr (is_channels_last) {
          plane = inputIndex % numPlane_;
          input_hw_index = ((inputIndex % in_n_stride_) - plane) / numPlane_;
        } else {
          plane = inputIndex / in_cf_c_stride_ % numPlane_;
          input_hw_index = ((inputIndex % in_n_stride_)) % in_cf_c_stride_;
        }
        int inputW = input_hw_index % gradInputSizeW_;
        int inputH = input_hw_index / gradInputSizeW_;
        int phstart =
            p_start(inputH, pad_h_, kernel_h_, dilation_h_, stride_h_);
        int phend = p_end(inputH, pad_h_, gradOutputSizeH_, stride_h_);
        int pwstart =
            p_start(inputW, pad_w_, kernel_w_, dilation_w_, stride_w_);
        int pwend = p_end(inputW, pad_w_, gradOutputSizeW_, stride_w_);
        if constexpr (is_channels_last) {
          int offset = batch * out_n_stride_ + plane;
          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              if (indices_[offset + (ph * gradOutputSizeW_ + pw) * numPlane_] ==
                  input_hw_index) {
                gradInput_[inputIndex] += static_cast<scalar_t>(
                    gradOutput_
                        [offset + (ph * gradOutputSizeW_ + pw) * numPlane_]);
              }
            }
          }
        } else {
          int offset = batch * out_n_stride_ + plane * out_cf_c_stride_;
          for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
              if (indices_[offset + ph * gradOutputSizeW_ + pw] ==
                  input_hw_index) {
                gradInput_[inputIndex] += static_cast<scalar_t>(
                    gradOutput_[offset + ph * gradOutputSizeW_ + pw]);
              }
            }
          }
        }
      }
    } while (cfg_.next(item, desc));
  }
  MaxPool2dBackwardOutDeterministicKernelFunctor(
      scalar_t* gradInput,
      scalar_t* gradOutput,
      int64_t* indices,
      int numPlane,
      int gradInputSizeH,
      int gradInputSizeW,
      int gradOutputSizeH,
      int gradOutputSizeW,
      int64_t gradInputSize,
      int out_cf_c_stride,
      int in_cf_c_stride,
      int out_n_stride,
      int in_n_stride,
      int kernel_h,
      int kernel_w,
      int stride_h,
      int stride_w,
      int pad_h,
      int pad_w,
      int dilation_h,
      int dilation_w,
      BatchKernelConfig cfg)
      : gradInput_(gradInput),
        gradOutput_(gradOutput),
        indices_(indices),
        numPlane_(numPlane),
        gradInputSizeH_(gradInputSizeH),
        gradInputSizeW_(gradInputSizeW),
        gradOutputSizeH_(gradOutputSizeH),
        gradOutputSizeW_(gradOutputSizeW),
        gradInputSize_(gradInputSize),
        out_cf_c_stride_(out_cf_c_stride),
        in_cf_c_stride_(in_cf_c_stride),
        out_n_stride_(out_n_stride),
        in_n_stride_(in_n_stride),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_h_(pad_h),
        pad_w_(pad_w),
        dilation_h_(dilation_h),
        dilation_w_(dilation_w),
        cfg_(cfg) {}

 private:
  scalar_t* gradInput_;
  scalar_t* gradOutput_;
  int64_t* indices_;
  int numPlane_;
  int gradInputSizeH_;
  int gradInputSizeW_;
  int gradOutputSizeH_;
  int gradOutputSizeW_;
  int64_t gradInputSize_;
  int out_cf_c_stride_;
  int in_cf_c_stride_;
  int out_n_stride_;
  int in_n_stride_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int dilation_h_;
  int dilation_w_;
  BatchKernelConfig cfg_;
};

template <typename scalar_t, bool is_channels_last>
void max_pool2d_backward_out_frame(
    scalar_t* gradInput,
    scalar_t* gradOutput,
    int64_t* indices,
    int numBatch,
    int numPlane,
    int gradInputSizeH,
    int gradInputSizeW,
    int gradOutputSizeH,
    int gradOutputSizeW,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int64_t gradOutputSize =
      numBatch * numPlane * gradOutputSizeH * gradOutputSizeW;
  int64_t gradInputSize = numBatch * numPlane * gradInputSizeH * gradInputSizeW;
  auto out_cf_c_stride = gradOutputSizeH * gradOutputSizeW;
  auto in_cf_c_stride = gradInputSizeH * gradInputSizeW;
  auto out_n_stride = numPlane * out_cf_c_stride;
  auto in_n_stride = numPlane * in_cf_c_stride;
  if (globalContext().deterministicAlgorithms() ||
      std::is_same_v<scalar_t, at::Half> ||
      std::is_same_v<scalar_t, at::BFloat16>) {
    BatchKernelConfig cfg = {
        1, gradInputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive};
    auto caller = MaxPool2dBackwardOutDeterministicKernelFunctor<
        scalar_t,
        is_channels_last>(
        gradInput,
        gradOutput,
        indices,
        numPlane,
        gradInputSizeH,
        gradInputSizeW,
        gradOutputSizeH,
        gradOutputSizeW,
        gradInputSize,
        out_cf_c_stride,
        in_cf_c_stride,
        out_n_stride,
        in_n_stride,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        cfg);
    sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, caller);
  } else {
    BatchKernelConfig cfg = {
        1, gradOutputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive};
    auto caller = MaxPool2dBackwardOutKernelFunctor<scalar_t, is_channels_last>(
        gradInput,
        gradOutput,
        indices,
        numPlane,
        gradInputSizeH,
        gradInputSizeW,
        gradOutputSizeH,
        gradOutputSizeW,
        gradOutputSize,
        out_cf_c_stride,
        in_cf_c_stride,
        out_n_stride,
        in_n_stride,
        cfg);
    sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, caller);
  }
}

Tensor& max_pool2d_with_indices_backward_out_kernel(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  NoNamesGuard guard;
  TensorArg gradInput_arg{gradInput, "gradInput", 1};
  TensorArg gradOutput_arg{gradOutput, "gradOutput", 2};
  TensorArg input_arg{input, "input", 3};
  TensorArg indices_arg{indices, "indices", 4};
  checkAllSameGPU(
      __func__, {gradInput_arg, gradOutput_arg, input_arg, indices_arg});

  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  int64_t outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  int64_t outputWidth = pooling_output_shape<int64_t>(
      inputWidth, kW, padW, dW, dilationW, ceil_mode);

  auto memory_format = input.suggest_memory_format();
  max_pool2d_backward_shape_check(
      input,
      gradOutput,
      indices,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      nInputPlane,
      inputHeight,
      inputWidth,
      outputHeight,
      outputWidth,
      memory_format);

  auto gradOutput_ = gradOutput.contiguous(memory_format);
  gradInput.zero_();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gradOutput.scalar_type(),
      "max_pool2d_backward_out_frame",
      [&] {
        switch (input.suggest_memory_format()) {
          case at::MemoryFormat::ChannelsLast:
            max_pool2d_backward_out_frame<scalar_t, true>(
                gradInput.data_ptr<scalar_t>(),
                gradOutput_.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                dilationH,
                dilationW);
            break;
          case at::MemoryFormat::Contiguous:
            max_pool2d_backward_out_frame<scalar_t, false>(
                gradInput.data_ptr<scalar_t>(),
                gradOutput_.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                nbatch,
                nInputPlane,
                inputHeight,
                inputWidth,
                outputHeight,
                outputWidth,
                kH,
                kW,
                dH,
                dW,
                padH,
                padW,
                dilationH,
                dilationW);
            break;
          default:
            TORCH_CHECK(
                false,
                "Unsupported memory format. Supports only ChannelsLast, Contiguous");
        }
      });
  return gradInput;
}

} // namespace xpu
} // namespace native
} // namespace at

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
