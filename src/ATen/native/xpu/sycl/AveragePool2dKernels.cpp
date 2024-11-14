#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <ATen/native/Pool.h>

#include <ATen/native/xpu/sycl/AveragePool2dKernels.h>
#include <comm/Runtime.h>
#include <comm/SYCLContext.h>
#include <comm/SYCLHelpers.h>

#include <ATen/native/xpu/sycl/AveragePool2dKernels.h>

namespace at::native {
namespace xpu {

inline int min(int a, int b) {
  return a <= b ? a : b;
}

inline int max(int a, int b) {
  return a >= b ? a : b;
}

template <typename scalar_t, typename accscalar_t>
struct AvgPool2dKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();

    if (index < total_elements_) {
      const int pw = index % pooled_width_;
      const int ph = (index / pooled_width_) % pooled_height_;
      const int c = (index / pooled_width_ / pooled_height_) % channels_;
      const int n = index / pooled_width_ / pooled_height_ / channels_;

      int hstart = ph * stride_h_ - pad_h_;
      int wstart = pw * stride_w_ - pad_w_;
      int hend = min(hstart + kernel_h_, height_ + pad_h_);
      int wend = min(wstart + kernel_w_, width_ + pad_w_);
      const int pool_size = (hend - hstart) * (wend - wstart);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
      hend = min(hend, height_);
      wend = min(wend, width_);

      if (hstart >= hend || wstart >= wend) {
        top_data_[index] = scalar_t(0);
        return;
      }

      accscalar_t aveval = accscalar_t(0);
      const scalar_t* const bottom_slice =
          bottom_data_ + (n * channels_ + c) * height_ * width_;

      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_slice[h * width_ + w];
        }
      }
      int divide_factor;
      if (use_divisor_) {
        divide_factor = divisor_override_;
      } else {
        if (count_include_pad_) {
          divide_factor = pool_size;
        } else {
          divide_factor = (hend - hstart) * (wend - wstart);
        }
      }
      top_data_[index] = static_cast<scalar_t>(aveval / divide_factor);
    }
  }
  AvgPool2dKernelFunctor(
      scalar_t* top_data,
      const scalar_t* bottom_data,
      int64_t total_elements,
      int64_t channels,
      int64_t height,
      int64_t width,
      int pooled_height,
      int pooled_width,
      int kernel_h,
      int kernel_w,
      int stride_h,
      int stride_w,
      int pad_h,
      int pad_w,
      int divisor_override,
      bool count_include_pad,
      bool use_divisor)
      : top_data_(top_data),
        bottom_data_(bottom_data),
        total_elements_(total_elements),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_h_(pad_h),
        pad_w_(pad_w),
        divisor_override_(divisor_override),
        count_include_pad_(count_include_pad),
        use_divisor_(use_divisor) {}

 private:
  scalar_t* top_data_;
  const scalar_t* bottom_data_;
  int64_t total_elements_;
  int64_t channels_;
  int64_t height_;
  int64_t width_;
  int pooled_height_;
  int pooled_width_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int divisor_override_;
  bool count_include_pad_;
  bool use_divisor_;
};

template <typename scalar_t, typename accscalar_t>
struct AvgPool2dChannelsLastKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    auto index = item.get_global_linear_id();

    if (index < total_elements_) {
      const int c = index % channels_;
      const int pw = (index / channels_) % pooled_width_;
      const int ph = (index / channels_ / pooled_width_) % pooled_height_;
      const int n = index / channels_ / pooled_width_ / pooled_height_;
      int hstart = ph * stride_h_ - pad_h_;
      int wstart = pw * stride_w_ - pad_w_;
      int hend = min(hstart + kernel_h_, height_ + pad_h_);
      int wend = min(wstart + kernel_w_, width_ + pad_w_);
      const int pool_size = (hend - hstart) * (wend - wstart);
      hstart = max(hstart, 0);
      wstart = max(wstart, 0);
      hend = min(hend, height_);
      wend = min(wend, width_);

      if (hstart >= hend || wstart >= wend) {
        top_data_[index] = scalar_t(0);
        return;
      }

      accscalar_t aveval = accscalar_t(0);
      const scalar_t* const bottom_slice =
          bottom_data_ + n * channels_ * height_ * width_ + c;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          aveval += bottom_slice[(h * width_ + w) * channels_];
        }
      }
      int divide_factor;
      if (use_divisor_) {
        divide_factor = divisor_override_;
      } else {
        if (count_include_pad_) {
          divide_factor = pool_size;
        } else {
          divide_factor = (hend - hstart) * (wend - wstart);
        }
      }
      top_data_[index] = static_cast<scalar_t>(aveval / divide_factor);
    }
  }
  AvgPool2dChannelsLastKernelFunctor(
      scalar_t* top_data,
      const scalar_t* bottom_data,
      int64_t total_elements,
      int64_t channels,
      int64_t height,
      int64_t width,
      int pooled_height,
      int pooled_width,
      int kernel_h,
      int kernel_w,
      int stride_h,
      int stride_w,
      int pad_h,
      int pad_w,
      int divisor_override,
      bool count_include_pad,
      bool use_divisor)
      : top_data_(top_data),
        bottom_data_(bottom_data),
        total_elements_(total_elements),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_h_(pad_h),
        pad_w_(pad_w),
        divisor_override_(divisor_override),
        count_include_pad_(count_include_pad),
        use_divisor_(use_divisor) {}

 private:
  scalar_t* top_data_;
  const scalar_t* bottom_data_;
  int64_t total_elements_;
  int64_t channels_;
  int64_t height_;
  int64_t width_;
  int pooled_height_;
  int pooled_width_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int divisor_override_;
  bool count_include_pad_;
  bool use_divisor_;
};

template <typename scalar_t, typename accscalar_t>
void launch_avg_pool2d_channels_last_kernel(
    const int total_elements,
    const Tensor& input,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& output,
    const int divisor_override,
    const bool count_include_pad,
    const bool use_divisor) {
  scalar_t* top_data = output.mutable_data_ptr<scalar_t>();
  const scalar_t* bottom_data = input.const_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const uint32_t group_size = static_cast<int>(syclMaxWorkItemsPerEU());
  const uint32_t global_range =
      ceil_div<uint32_t>(total_elements, group_size) * group_size;

  auto kfn = AvgPool2dChannelsLastKernelFunctor<scalar_t, accscalar_t>(
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
  sycl_kernel_submit(global_range, group_size, queue, kfn);
}

template <typename scalar_t, typename accscalar_t>
void launch_avg_pool2d_kernel(
    const int total_elements,
    const Tensor& input,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& output,
    const int divisor_override,
    const bool count_include_pad,
    const bool use_divisor) {
  scalar_t* top_data = output.mutable_data_ptr<scalar_t>();
  const scalar_t* bottom_data = input.const_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const uint32_t group_size = static_cast<int>(syclMaxWorkItemsPerEU());
  const uint32_t global_range =
      ceil_div<uint32_t>(total_elements, group_size) * group_size;

  auto kfn = AvgPool2dKernelFunctor<scalar_t, accscalar_t>(
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
  sycl_kernel_submit(global_range, group_size, queue, kfn);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool2dChannelsLastBackwardKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    index_t index = item.get_global_linear_id();
    if (index < total_elements_) {
      const int c = index % channels_;
      const int w = (index / channels_) % width_ + pad_w_;
      const int h = (index / channels_ / width_) % height_ + pad_h_;
      const int n = index / channels_ / width_ / height_;
      const int phstart = (h < kernel_h_) ? 0 : (h - kernel_h_) / stride_h_ + 1;
      const int phend = min(h / stride_h_ + 1, pooled_height_);
      const int pwstart = (w < kernel_w_) ? 0 : (w - kernel_w_) / stride_w_ + 1;
      const int pwend = min(w / stride_w_ + 1, pooled_width_);
      accscalar_t gradient = accscalar_t(0);
      const scalar_t* const top_slice =
          top_data_ + n * channels_ * pooled_height_ * pooled_width_ + c;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int hstart = ph * stride_h_ - pad_h_;
          int wstart = pw * stride_w_ - pad_w_;
          int hend = min(hstart + kernel_h_, height_ + pad_h_);
          int wend = min(wstart + kernel_w_, width_ + pad_w_);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, height_);
          wend = min(wend, width_);
          if (hstart >= hend || wstart >= wend) {
            continue;
          }
          int divide_factor;
          if (use_divisor_) {
            divide_factor = divisor_override_;
          } else {
            if (count_include_pad_) {
              divide_factor = pool_size;
            } else {
              divide_factor = (hend - hstart) * (wend - wstart);
            }
          }
          gradient +=
              top_slice[(ph * pooled_width_ + pw) * channels_] / divide_factor;
        }
      }
      bottom_data_[index] = static_cast<scalar_t>(gradient);
    }
  }
  AvgPool2dChannelsLastBackwardKernelFunctor(
      const scalar_t* top_data,
      scalar_t* bottom_data,
      int64_t total_elements,
      int64_t channels,
      int64_t height,
      int64_t width,
      int pooled_height,
      int pooled_width,
      int kernel_h,
      int kernel_w,
      int stride_h,
      int stride_w,
      int pad_h,
      int pad_w,
      int divisor_override,
      bool count_include_pad,
      bool use_divisor)
      : top_data_(top_data),
        bottom_data_(bottom_data),
        total_elements_(total_elements),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_h_(pad_h),
        pad_w_(pad_w),
        divisor_override_(divisor_override),
        count_include_pad_(count_include_pad),
        use_divisor_(use_divisor) {}

 private:
  const scalar_t* top_data_;
  scalar_t* bottom_data_;
  int64_t total_elements_;
  int64_t channels_;
  int64_t height_;
  int64_t width_;
  int pooled_height_;
  int pooled_width_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int divisor_override_;
  bool count_include_pad_;
  bool use_divisor_;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
struct AvgPool2dBackwarKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    index_t index = item.get_global_linear_id();
    if (index < total_elements_) {
      // find out the local index
      // find out the local offset
      const int w = index % width_ + pad_w_;
      const int h = (index / width_) % height_ + pad_h_;
      const int c = (index / width_ / height_) % channels_;
      const int n = index / width_ / height_ / channels_;
      const int phstart = (h < kernel_h_) ? 0 : (h - kernel_h_) / stride_h_ + 1;
      const int phend = min(h / stride_h_ + 1, pooled_height_);
      const int pwstart = (w < kernel_w_) ? 0 : (w - kernel_w_) / stride_w_ + 1;
      const int pwend = min(w / stride_w_ + 1, pooled_width_);
      accscalar_t gradient = accscalar_t(0);
      const scalar_t* const top_data_slice =
          top_data_ + (n * channels_ + c) * pooled_height_ * pooled_width_;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          // figure out the pooling size
          int hstart = ph * stride_h_ - pad_h_;
          int wstart = pw * stride_w_ - pad_w_;
          int hend = min(hstart + kernel_h_, height_ + pad_h_);
          int wend = min(wstart + kernel_w_, width_ + pad_w_);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = max(hstart, 0);
          wstart = max(wstart, 0);
          hend = min(hend, height_);
          wend = min(wend, width_);
          if (hstart >= hend || wstart >= wend) {
            continue;
          }
          int divide_factor;
          if (use_divisor_) {
            divide_factor = divisor_override_;
          } else {
            if (count_include_pad_) {
              divide_factor = pool_size;
            } else {
              divide_factor = (hend - hstart) * (wend - wstart);
            }
          }
          gradient += top_data_slice[ph * pooled_width_ + pw] / divide_factor;
        }
      }
      bottom_data_[index] = static_cast<scalar_t>(gradient);
    }
  }
  AvgPool2dBackwarKernelFunctor(
      const scalar_t* top_data,
      scalar_t* bottom_data,
      int64_t total_elements,
      int64_t channels,
      int64_t height,
      int64_t width,
      int pooled_height,
      int pooled_width,
      int kernel_h,
      int kernel_w,
      int stride_h,
      int stride_w,
      int pad_h,
      int pad_w,
      int divisor_override,
      bool count_include_pad,
      bool use_divisor)
      : top_data_(top_data),
        bottom_data_(bottom_data),
        total_elements_(total_elements),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        kernel_h_(kernel_h),
        kernel_w_(kernel_w),
        stride_h_(stride_h),
        stride_w_(stride_w),
        pad_h_(pad_h),
        pad_w_(pad_w),
        divisor_override_(divisor_override),
        count_include_pad_(count_include_pad),
        use_divisor_(use_divisor) {}

 private:
  const scalar_t* top_data_;
  scalar_t* bottom_data_;
  int64_t total_elements_;
  int64_t channels_;
  int64_t height_;
  int64_t width_;
  int pooled_height_;
  int pooled_width_;
  int kernel_h_;
  int kernel_w_;
  int stride_h_;
  int stride_w_;
  int pad_h_;
  int pad_w_;
  int divisor_override_;
  bool count_include_pad_;
  bool use_divisor_;
};

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_backward_channels_last_kernel(
    const index_t total_elements,
    const Tensor& grad_output,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& grad_input,
    const int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  const scalar_t* top_data = grad_output.const_data_ptr<scalar_t>();
  scalar_t* bottom_data = grad_input.mutable_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const uint32_t group_size = static_cast<int>(syclMaxWorkItemsPerEU());
  const uint32_t global_range =
      ceil_div<uint32_t>(total_elements, group_size) * group_size;

  auto kfn = AvgPool2dChannelsLastBackwardKernelFunctor<
      scalar_t,
      accscalar_t,
      index_t>(
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
  sycl_kernel_submit(global_range, group_size, queue, kfn);
}

template <typename scalar_t, typename accscalar_t, typename index_t>
void launch_avg_pool2d_backward_kernel(
    const index_t total_elements,
    const Tensor& grad_output,
    const int64_t channels,
    const int64_t height,
    const int64_t width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const Tensor& grad_input,
    const int divisor_override,
    bool count_include_pad,
    bool use_divisor) {
  const scalar_t* top_data = grad_output.const_data_ptr<scalar_t>();
  scalar_t* bottom_data = grad_input.mutable_data_ptr<scalar_t>();

  auto& queue = at::xpu::getCurrentSYCLQueue();
  const uint32_t group_size = static_cast<int>(syclMaxWorkItemsPerEU());
  const uint32_t global_range =
      ceil_div<uint32_t>(total_elements, group_size) * group_size;

  auto kfn = AvgPool2dBackwarKernelFunctor<scalar_t, accscalar_t, index_t>(
      top_data,
      bottom_data,
      total_elements,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      kernel_h,
      kernel_w,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      divisor_override,
      count_include_pad,
      use_divisor);
  sycl_kernel_submit(global_range, group_size, queue, kfn);
}

void avg_pool2d_kernel(
    const Tensor& input_,
    int64_t kH_,
    int64_t kW_,
    int64_t dH_,
    int64_t dW_,
    int64_t padH_,
    int64_t padW_,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    const Tensor& output) {
  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW_, padW_, dW_, 1, ceil_mode);
  int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH_, padH_, dH_, 1, ceil_mode);
  const auto memory_format = input_.suggest_memory_format();

  Tensor input = input_.contiguous(memory_format);
  const auto count = safe_downcast<int32_t, int64_t>(output.numel());

  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value =
      use_divisor ? divisor_override.value() : 0;
  if (count != 0) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, input.scalar_type(), "avg_pool2d_xpu", [&] {
          using accscalar_t = acc_type_device<scalar_t, kXPU>;

          switch (memory_format) {
            case MemoryFormat::ChannelsLast: {
              output.unsafeGetTensorImpl()->empty_tensor_restride(
                  MemoryFormat::ChannelsLast);
              launch_avg_pool2d_channels_last_kernel<scalar_t, accscalar_t>(
                  count,
                  input,
                  nInputPlane,
                  inputHeight,
                  inputWidth,
                  outputHeight,
                  outputWidth,
                  kH_,
                  kW_,
                  dH_,
                  dW_,
                  padH_,
                  padW_,
                  output,
                  divisor_override_value,
                  count_include_pad,
                  use_divisor);
              break;
            }
            case MemoryFormat::Contiguous: {
              launch_avg_pool2d_kernel<scalar_t, accscalar_t>(
                  count,
                  input,
                  nInputPlane,
                  inputHeight,
                  inputWidth,
                  outputHeight,
                  outputWidth,
                  kH_,
                  kW_,
                  dH_,
                  dW_,
                  padH_,
                  padW_,
                  output,
                  divisor_override_value,
                  count_include_pad,
                  use_divisor);
              break;
            }
            default:
              TORCH_CHECK(
                  false,
                  "Unsupported memory format. Supports only "
                  "ChannelsLast, Contiguous");
          }
        });
  }
}

void avg_pool2d_backward_kernel(
    const Tensor& gradOutput_,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    const Tensor& gradInput) {
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

  const auto memory_format = input_.suggest_memory_format();
  const Tensor input = input_.contiguous(memory_format);
  const Tensor gradOutput = gradOutput_.contiguous(memory_format);

  const int64_t nInputPlane = input_.size(-3);
  const int64_t inputHeight = input_.size(-2);
  const int64_t inputWidth = input_.size(-1);

  int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  const auto count = input.numel();
  if (count == 0) {
    return;
  }
  bool use_divisor = divisor_override.has_value();
  const auto divisor_override_value =
      use_divisor ? divisor_override.value() : 0;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input.scalar_type(), "avg_pool2d_backward_xpu", [&] {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;

        AT_DISPATCH_INDEX_TYPES(
            at::native::canUse32BitIndexMath(input, INT_MAX) ? ScalarType::Int
                                                             : ScalarType::Long,
            "avg_pool2d_backward_xpu",
            [&] {
              switch (memory_format) {
                case MemoryFormat::ChannelsLast: {
                  gradInput.unsafeGetTensorImpl()->empty_tensor_restride(
                      MemoryFormat::ChannelsLast);
                  launch_avg_pool2d_backward_channels_last_kernel<
                      scalar_t,
                      accscalar_t,
                      index_t>(
                      count,
                      gradOutput,
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
                      gradInput,
                      divisor_override_value,
                      count_include_pad,
                      use_divisor);
                  break;
                }
                case MemoryFormat::Contiguous: {
                  launch_avg_pool2d_backward_kernel<
                      scalar_t,
                      accscalar_t,
                      index_t>(
                      count,
                      gradOutput,
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
                      gradInput,
                      divisor_override_value,
                      count_include_pad,
                      use_divisor);
                  break;
                }
                default:
                  TORCH_CHECK(
                      false,
                      "Unsupported memory format. Supports only "
                      "ChannelsLast, Contiguous");
              }
            });
      });
}

} // namespace xpu
} // namespace at::native
