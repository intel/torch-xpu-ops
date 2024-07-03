#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/AccumulateType.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/BatchKernel.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

namespace at::native::xpu {

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
struct MaxPool2dKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);

    do {
      if (desc.glb_problem < cfg_.problem_) {
        int outputIndex = desc.glb_problem;
        int batch = outputIndex / stride_;
        int plane, outputH, outputW;
        int64_t load_offset, store_offset;
        if constexpr (is_channels_last) {
          plane = outputIndex % numPlane_;
          outputH = outputIndex / numPlane_ / outputSizeW_ % outputSizeH_;
          outputW = outputIndex / numPlane_ % outputSizeW_;
          store_offset = batch * outputSizeH_ * outputSizeW_ * numPlane_ +
              plane + outputH * outputSizeW_ * numPlane_ + outputW * numPlane_;
        } else {
          plane = (outputIndex / outputSizeH_ / outputSizeW_) % numPlane_;
          outputH = outputIndex / outputSizeW_ % outputSizeH_;
          outputW = outputIndex % outputSizeW_;
          store_offset = batch * numPlane_ * outputSizeH_ * outputSizeW_ +
              plane * outputSizeH_ * outputSizeW_ + outputH * outputSizeW_ +
              outputW;
        }
        scalar_t maxVal = at::numeric_limits<scalar_t>::lower_bound();
        int maxIndex = -1;
        int StartH = outputH * dH_ - padH_;
        int StartW = outputW * dW_ - padW_;
        int EndH = std::min(StartH + (kH_ - 1) * dilationH_ + 1, inputSizeH_);
        int EndW = std::min(StartW + (kW_ - 1) * dilationW_ + 1, inputSizeW_);
        while (StartH < 0)
          StartH += dilationH_;
        while (StartW < 0)
          StartW += dilationW_;
#pragma unroll
        for (int h = StartH; h < EndH; h += dilationH_) {
#pragma unroll
          for (int w = StartW; w < EndW; w += dilationW_) {
            if constexpr (is_channels_last) {
              load_offset = batch * inputSizeH_ * inputSizeW_ * numPlane_ +
                  plane + h * inputSizeW_ * numPlane_ + w * numPlane_;
            } else {
              load_offset = batch * numPlane_ * inputSizeH_ * inputSizeW_ +
                  plane * inputSizeH_ * inputSizeW_ + h * inputSizeW_ + w;
            }
            scalar_t val = input_[load_offset];
            if ((static_cast<scalar_t>(val) > maxVal) || at::_isnan(val)) {
              maxIndex = h * inputSizeW_ + w;
              maxVal = static_cast<scalar_t>(val);
            }
          }
        }
        indices_[store_offset] = maxIndex;
        output_[store_offset] = static_cast<scalar_t>(maxVal);
      }
    } while (cfg_.next(item, desc));
  }
  MaxPool2dKernelFunctor(
      scalar_t* output,
      int64_t* indices,
      scalar_t* input,
      int numPlane,
      int inputSizeH,
      int inputSizeW,
      int outputSizeH,
      int outputSizeW,
      int kH,
      int kW,
      int dH,
      int dW,
      int padH,
      int padW,
      int dilationH,
      int dilationW,
      int stride,
      BatchKernelConfig cfg)
      : output_(output),
        indices_(indices),
        input_(input),
        numPlane_(numPlane),
        inputSizeH_(inputSizeH),
        inputSizeW_(inputSizeW),
        outputSizeH_(outputSizeH),
        outputSizeW_(outputSizeW),
        kH_(kH),
        kW_(kW),
        dH_(dH),
        dW_(dW),
        padH_(padH),
        padW_(padW),
        dilationH_(dilationH),
        dilationW_(dilationW),
        stride_(stride),
        cfg_(cfg) {}

 private:
  scalar_t* output_;
  int64_t* indices_;
  scalar_t* input_;
  int numPlane_;
  int inputSizeH_;
  int inputSizeW_;
  int outputSizeH_;
  int outputSizeW_;
  int kH_;
  int kW_;
  int dH_;
  int dW_;
  int padH_;
  int padW_;
  int dilationH_;
  int dilationW_;
  int stride_;
  BatchKernelConfig cfg_;
};

template <typename scalar_t, bool is_channels_last>
struct MaxPool2dBackwardKernelFunctor {
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
  MaxPool2dBackwardKernelFunctor(
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
struct MaxPool2dBackwardDeterministicKernelFunctor {
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
  MaxPool2dBackwardDeterministicKernelFunctor(
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
void launch_max_pool2d_kernel(
    scalar_t* output,
    int64_t* indices,
    scalar_t* input,
    int numBatch,
    int numPlane,
    int inputSizeH,
    int inputSizeW,
    int outputSizeH,
    int outputSizeW,
    int kH,
    int kW,
    int dH,
    int dW,
    int padH,
    int padW,
    int dilationH,
    int dilationW) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int outputSize = numBatch * numPlane * outputSizeH * outputSizeW;
  int stride = numPlane * outputSizeH * outputSizeW;
  BatchKernelConfig cfg = {
      1, outputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive};
  auto kfn = MaxPool2dKernelFunctor<scalar_t, is_channels_last>(
      output,
      indices,
      input,
      numPlane,
      inputSizeH,
      inputSizeW,
      outputSizeH,
      outputSizeW,
      kH,
      kW,
      dH,
      dW,
      padH,
      padW,
      dilationH,
      dilationW,
      stride,
      cfg);
  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
}

template <typename scalar_t, bool is_channels_last>
void launch_max_pool2d_backward_kernel(
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
    auto kfn =
        MaxPool2dBackwardDeterministicKernelFunctor<scalar_t, is_channels_last>(
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
    sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
  } else {
    BatchKernelConfig cfg = {
        1, gradOutputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive};
    auto kfn = MaxPool2dBackwardKernelFunctor<scalar_t, is_channels_last>(
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
    sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
  }
}

void max_pool2d_with_indices_kernel(
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    const Tensor& output_,
    const Tensor& indices_) {
  NoNamesGuard guard;

  TensorArg output_arg{output_, "output", 1};
  TensorArg indices_arg{indices_, "indices", 2};
  TensorArg input_arg{input_, "input_", 3};

  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});
  if (output_.numel() == 0) {
    return;
  }

  auto smf = input_.suggest_memory_format();
  bool is_3d = input_.ndimension() == 3;
  Tensor input, indices, output;
  if (is_3d) {
    input = input_.contiguous();
    indices = indices_.contiguous();
    output = output_.contiguous();
  } else {
    input = input_.contiguous(smf);
    indices = indices_.contiguous(smf);
    output = output_.contiguous(smf);
  }

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

  const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  const int64_t outputHeight = output.size(-2);
  const int64_t outputWidth = output.size(-1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "max_pool2d_xpu", [&] {
        switch (smf) {
          case MemoryFormat::ChannelsLast: {
            launch_max_pool2d_kernel<scalar_t, true>(
                output.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                input.data_ptr<scalar_t>(),
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
          }
          case MemoryFormat::Contiguous: {
            launch_max_pool2d_kernel<scalar_t, false>(
                output.data_ptr<scalar_t>(),
                indices.data_ptr<int64_t>(),
                input.data_ptr<scalar_t>(),
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
          }
          default:
            TORCH_CHECK(
                false,
                "Unsupported memory format. Supports only ChannelsLast, Contiguous");
        }
      });

  if ((is_3d && !indices_.is_contiguous()) ||
      (!is_3d && !indices_.is_contiguous(smf))) {
    indices_.copy_(indices);
  }

  if ((is_3d && !output_.is_contiguous()) ||
      (!is_3d && !output_.is_contiguous(smf))) {
    output_.copy_(output);
  }
}

void max_pool2d_with_indices_backward_kernel(
    const Tensor& gradInput_,
    const Tensor& gradOutput_,
    const Tensor& input_,
    const Tensor& indices_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  NoNamesGuard guard;
  TensorArg gradInput_arg{gradInput_, "gradInput", 1};
  TensorArg gradOutput_arg{gradOutput_, "gradOutput", 2};
  TensorArg input_arg{input_, "input", 3};
  TensorArg indices_arg{indices_, "indices", 4};
  checkAllSameGPU(
      __func__, {gradInput_arg, gradOutput_arg, input_arg, indices_arg});

  auto smf = input_.suggest_memory_format();
  bool is_3d = input_.ndimension() == 3;
  Tensor input, gradOutput, indices, gradInput;
  if (is_3d) {
    input = input_.contiguous();
    gradOutput = gradOutput_.contiguous();
    indices = indices_.contiguous();
    gradInput = gradInput_.contiguous();
  } else {
    input = input_.contiguous(smf);
    gradOutput = gradOutput_.contiguous(smf);
    indices = indices_.contiguous(smf);
    gradInput = gradInput_.contiguous(smf);
  }
  gradInput.zero_();

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

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gradOutput.scalar_type(),
      "max_pool2d_backward_xpu",
      [&] {
        switch (smf) {
          case at::MemoryFormat::ChannelsLast:
            launch_max_pool2d_backward_kernel<scalar_t, true>(
                gradInput.data_ptr<scalar_t>(),
                gradOutput.data_ptr<scalar_t>(),
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
            launch_max_pool2d_backward_kernel<scalar_t, false>(
                gradInput.data_ptr<scalar_t>(),
                gradOutput.data_ptr<scalar_t>(),
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

  if ((is_3d && !gradInput_.is_contiguous()) ||
      (!is_3d && !gradInput_.is_contiguous(smf))) {
    gradInput_.copy_(gradInput);
  }
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
