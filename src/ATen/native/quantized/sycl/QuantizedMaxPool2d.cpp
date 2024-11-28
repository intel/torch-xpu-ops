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

#include <ATen/native/quantized/sycl/QuantizedMaxPool2d.h>
namespace at::native::xpu {

namespace {
void check_maxpool2d_params(
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation) {
  TORCH_CHECK(
      kernel_size.size() == 1 || kernel_size.size() == 2,
      "Expected 1d or 2d kernel size, got ",
      kernel_size.size());
  TORCH_CHECK(
      stride.empty() || stride.size() == 2,
      "Expected no strides or 2d strides, got",
      stride.size());
  TORCH_CHECK(
      padding.size() == 1 || padding.size() == 2,
      "Expected 1d or 2d padding, got ",
      padding.size());
  TORCH_CHECK(
      dilation.size() == 1 || dilation.size() == 2,
      "Expected 1d or 2d dilation, got ",
      dilation.size());
}
} // anonymous namespace

template <typename scalar_t>
struct QuantizedMaxPool2dKernelFunctor {
  void operator()(sycl::nd_item<2> item) const {
    auto desc = cfg_.get_item_desc(item);

    do {
      if (desc.glb_problem < cfg_.problem_) {
        int idx = desc.glb_problem;
        int64_t b{0}, row{0}, col{0};
        b = idx / stride_;
        col = idx % oW_;
        row = idx / oW_ % oH_;

        int64_t output_base_offset = (b * oW_ * oH_ + row * oW_ + col) * iC_;

        // Get the boundary.
        int64_t h_start = row * sH_ - pH_;
        int64_t w_start = col * sW_ - pW_;
        int64_t h_end = std::min(h_start + (kH_ - 1) * dH_ + 1, iH_);
        int64_t w_end = std::min(w_start + (kW_ - 1) * dW_ + 1, iW_);
        while (h_start < 0)
          h_start += dH_;
        while (w_start < 0)
          w_start += dW_;

        // Stock pytorch's cpu implementation use vectorized instructions
        // through channels such as AVX-512. We use for-loop directly.
        int64_t w, h, c;
#pragma unroll
        for (c = 0; c < iC_; c++) {
          scalar_t maxVal = at::numeric_limits<scalar_t>::lower_bound();
#pragma unroll
          for (h = h_start; h < h_end; h += dH_) {
#pragma unroll
            for (w = w_start; w < w_end; w += dW_) {
              int64_t input_base_offset = (b * iW_ * iH_ + h * iW_ + w) * iC_;
              scalar_t val = input_[input_base_offset + c];
              if ((static_cast<scalar_t>(val) > maxVal) || at::_isnan(val)) {
                maxVal = static_cast<scalar_t>(val);
              }
            }
          }
          output_[output_base_offset + c] = static_cast<scalar_t>(maxVal);
        }
      }
    } while (cfg_.next(item, desc));
  }

  QuantizedMaxPool2dKernelFunctor(
      scalar_t* output,
      scalar_t* input,
      int64_t iC,
      int64_t iH,
      int64_t iW,
      int64_t oH,
      int64_t oW,
      int64_t kH,
      int64_t kW,
      int64_t sH,
      int64_t sW,
      int64_t pH,
      int64_t pW,
      int64_t dH,
      int64_t dW,
      int64_t stride,
      BatchKernelConfig cfg)
      : output_(output),
        input_(input),
        iC_(iC),
        iH_(iH),
        iW_(iW),
        oH_(oH),
        oW_(oW),
        kH_(kH),
        kW_(kW),
        sH_(sH),
        sW_(sW),
        pH_(pH),
        pW_(pW),
        dH_(dH),
        dW_(dW),
        stride_(stride),
        cfg_(cfg) {}

 private:
  scalar_t* output_;
  scalar_t* input_;
  int64_t iC_; // input/output channels
  int64_t iH_;
  int64_t iW_; // input sizes
  int64_t oH_;
  int64_t oW_; // output sizes
  int64_t kH_;
  int64_t kW_; // kernel size
  int64_t sH_;
  int64_t sW_; // strides
  int64_t pH_;
  int64_t pW_; // padding
  int64_t dH_;
  int64_t dW_; // dilation
  int64_t stride_;
  BatchKernelConfig cfg_;
};

template <typename scalar_t>
void launch_quantized_max_pool2d_kernel(
    scalar_t* output,
    scalar_t* input,
    int64_t nBatch,
    int64_t iC,
    int64_t iH,
    int64_t iW,
    int64_t oH,
    int64_t oW,
    int64_t kH,
    int64_t kW,
    int64_t sH,
    int64_t sW,
    int64_t pH,
    int64_t pW,
    int64_t dH,
    int64_t dW) {
  using KernelClass = QuantizedMaxPool2dKernelFunctor<scalar_t>;

  auto& queue = at::xpu::getCurrentSYCLQueue();
  int outputSize = nBatch * oH * oW;
  int stride = oH * oW;
  BatchKernelConfig cfg = BatchKernelConfig::make_config<KernelClass>(
      1, outputSize, 1, 1, true, BatchKernelConfig::Policy::pAdaptive);
  auto kfn = KernelClass(
      output,
      input,
      iC,
      iH,
      iW,
      oH,
      oW,
      kH,
      kW,
      sH,
      sW,
      pH,
      pW,
      dH,
      dW,
      stride,
      cfg);
  sycl_kernel_submit(cfg.global_size(), cfg.group_size(), queue, kfn);
}

Tensor quantized_max_pool2d_kernel(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  check_maxpool2d_params(kernel_size, stride, padding, dilation);
  if (stride.empty()) {
    stride = kernel_size;
  }
  Tensor output;
  int ndim = input.dim();
  int64_t kH = kernel_size[0];
  int64_t kW = kernel_size[1];
  int64_t sH = stride[0];
  int64_t sW = stride[1];
  int64_t pH = padding[0];
  int64_t pW = padding[1];
  int64_t dH = dilation[0];
  int64_t dW = dilation[1];

  // Check input dimensions.
  TORCH_CHECK(kH > 0 && kW > 0, "kernel_size should be greater than zero.");
  TORCH_CHECK(sH > 0 && sW > 0, "strides should be greater than zero.");
  TORCH_CHECK(
      dH > 0 && dW > 0,
      "dilation should be greater than zero. "
      "Got (",
      dH,
      ", ",
      dW,
      ")");
  TORCH_CHECK(
      ndim == 3 || ndim == 4, "Expecting the input tensor of rank 3 or 4.");

  int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
  int64_t iC = input.size(-3);
  int64_t iH = input.size(-2);
  int64_t iW = input.size(-1);
  int64_t oH = pooling_output_shape(iH, kH, pH, sH, dH, ceil_mode);
  int64_t oW = pooling_output_shape(iW, kW, pW, sW, dW, ceil_mode);
  int64_t oC = iC;

  TORCH_CHECK(
      oH > 0 && oW > 0,
      "Given input size: (",
      iC,
      "x",
      iH,
      "x",
      iW,
      "). Calculated output size: (",
      oC,
      "x",
      oH,
      "x",
      oW,
      "). Output size is too small.");

  std::vector<int64_t> oSizes;
  if (ndim == 3) {
    oSizes = {oC, oH, oW};
  } else {
    oSizes = {nbatch, oC, oH, oW};
  }

  // Create an input
  output = at::empty(
      oSizes,
      input.options()
          .device(c10::kXPU)
          .dtype(input.scalar_type())
          .memory_format(c10::MemoryFormat::ChannelsLast));

  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast)) {
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "quantized_max_pool2d_xpu", [&]() {
          launch_quantized_max_pool2d_kernel(
              output.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(),
              nbatch,
              iC,
              iH,
              iW,
              oH,
              oW,
              kH,
              kW,
              sH,
              sW,
              pH,
              pW,
              dH,
              dW);
        });
  } else {
    // If input is uint8 and contiguous memory format,
    // Use the channels_last implementation and convert output back to
    // contiguous.
    auto input_nhwc = input.contiguous(c10::MemoryFormat::ChannelsLast);
    AT_DISPATCH_INTEGRAL_TYPES(
        input.scalar_type(), "quantized_max_pool2d_xpu", [&]() {
          launch_quantized_max_pool2d_kernel(
              output.data_ptr<scalar_t>(),
              input_nhwc.data_ptr<scalar_t>(),
              nbatch,
              iC,
              iH,
              iW,
              oH,
              oW,
              kH,
              kW,
              sH,
              sW,
              pH,
              pW,
              dH,
              dW);
        });
    output = output.contiguous();
  }
  return output;
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
