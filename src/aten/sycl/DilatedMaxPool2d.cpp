#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"

#include <ATen/ATen.h>
#include <ATen/native/Pool.h>
#include <ATen/native/utils/ParamUtils.h>

#include <aten/sycl/BatchKernel.h>
#include <comm/Runtime.h>
#include <comm/SYCLHelpers.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, bool is_channels_last>
struct MaxPool2dBackwardOutFrameKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg.get_item_desc(item);

    do {
      if (desc.glb_problem < cfg.problem_) {
        int batch = desc.glb_problem / out_n_stride;
        int outputIndex = desc.glb_problem;
        if constexpr (is_channels_last) {
          int plane = outputIndex % numPlane;
          int64_t index = indices[outputIndex];
          int64_t gI_offset = batch * in_n_stride + plane + index * numPlane;
          atomicAdd(
              (sycl_global_ptr_pt<scalar_t>)&gradInput[gI_offset],
              gradOutput[outputIndex]);
        } else {
          int plane = outputIndex / out_cf_c_stride % numPlane;
          int64_t index = indices[outputIndex];
          int inputW = index % gradInputSizeW;
          int inputH = index / gradInputSizeW;
          int64_t gI_offset =
              batch * in_n_stride + plane * in_cf_c_stride + index;
          atomicAdd(
              (sycl_global_ptr_pt<scalar_t>)&gradInput[gI_offset],
              gradOutput[outputIndex]);
        }
      }
    } while (cfg.next(item, desc));
  }
  MaxPool2dBackwardOutFrameKernelFunctor(
      scalar_t* gradInput_,
      scalar_t* gradOutput_,
      int64_t* indices_,
      int numPlane_,
      int gradInputSizeH_,
      int gradInputSizeW_,
      int gradOutputSizeH_,
      int gradOutputSizeW_,
      int64_t gradOutputSize_,
      int out_cf_c_stride_,
      int in_cf_c_stride_,
      int out_n_stride_,
      int in_n_stride_,
      BatchKernelConfig cfg_)
      : gradInput(gradInput_),
        gradOutput(gradOutput_),
        indices(indices_),
        numPlane(numPlane_),
        gradInputSizeH(gradInputSizeH_),
        gradInputSizeW(gradInputSizeW_),
        gradOutputSizeH(gradOutputSizeH_),
        gradOutputSizeW(gradOutputSizeW_),
        gradOutputSize(gradOutputSize_),
        out_cf_c_stride(out_cf_c_stride_),
        in_cf_c_stride(in_cf_c_stride_),
        out_n_stride(out_n_stride_),
        in_n_stride(in_n_stride_),
        cfg(cfg_) {}

 private:
  scalar_t* gradInput;
  scalar_t* gradOutput;
  int64_t* indices;
  int numPlane;
  int gradInputSizeH;
  int gradInputSizeW;
  int gradOutputSizeH;
  int gradOutputSizeW;
  int64_t gradOutputSize;
  int out_cf_c_stride;
  int in_cf_c_stride;
  int out_n_stride;
  int in_n_stride;
  BatchKernelConfig cfg;
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
    int gradOutputSizeW) {
  auto& queue = at::xpu::getCurrentSYCLQueue();
  int64_t gradOutputSize =
      numBatch * numPlane * gradOutputSizeH * gradOutputSizeW;
  auto out_cf_c_stride = gradOutputSizeH * gradOutputSizeW;
  auto in_cf_c_stride = gradInputSizeH * gradInputSizeW;
  auto out_n_stride = numPlane * out_cf_c_stride;
  auto in_n_stride = numPlane * in_cf_c_stride;
  BatchKernelConfig cfg = {
      1, gradOutputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive};

  auto kfn = MaxPool2dBackwardOutFrameKernelFunctor<scalar_t, is_channels_last>(
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
} // namespace impl

Tensor& max_pool2d_with_indices_backward_out_template(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input,
    const Tensor& indices,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "max_pool2d: kernel_size must either be a single int, or a tuple "
      "of two ints")
  const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kW = kernel_size.size() == 1
      ? kH
      : safe_downcast<int, int64_t>(kernel_size[1]);

  TORCH_CHECK(
      stride.size() == 0 || stride.size() == 1 || stride.size() == 2,
      "max_pool2d: stride must either be omitted, a single int, or a "
      "tuple of two ints")
  const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
  const int dW = stride.empty() ? kW
      : stride.size() == 1      ? dH
                                : safe_downcast<int, int64_t>(stride[1]);

  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "max_pool2d: padding must be either be a single int, or a tuple "
      "of two ints");
  const int padH = safe_downcast<int, int64_t>(padding[0]);
  const int padW =
      padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "max_pool2d: dilation must be either a single int, or a tuple of "
      "two ints");
  const int dilationH = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationW = dilation.size() == 1
      ? dilationH
      : safe_downcast<int, int64_t>(dilation[1]);

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  const auto nInputPlane = input.size(-3);
  const auto inputHeight = input.size(-2);
  const auto inputWidth = input.size(-1);

  const int64_t dims = 2;
  auto kernel_size_vec =
      expand_param_if_needed(kernel_size, "kernel_size", dims);
  std::vector<int64_t> empty_stride_vec = {dH, dW};
  auto stride_vec = stride.empty()
      ? empty_stride_vec
      : expand_param_if_needed(stride, "stride", dims);
  auto padding_vec = expand_param_if_needed(padding, "padding", dims);

  auto padding_vec_l = padding_vec;
  auto padding_vec_r = padding_vec;
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

  std::vector<int64_t> output_sizes;
  int64_t outputHeight, outputWidth;

  // Unsqueeze 3D input(C, H, W) -> 4D input(1, C, H, W)
  auto input_4d = (input.ndimension() == 3) ? (input.unsqueeze(0)) : (input);

  outputHeight = pooling_output_shape<int64_t>(
      inputHeight, kH, padH, dH, dilationH, ceil_mode);
  outputWidth = pooling_output_shape<int64_t>(
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
                outputWidth);
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
                outputWidth);
            break;
        }
      });
  return gradInput;
}

} // namespace xpu
} // namespace native
} // namespace at