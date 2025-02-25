#pragma clang diagnostic push
#pragma GCC diagnostic push
// Avoid SYCL compiler return-type error
#pragma clang diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wreturn-type"
#include <ATen/ceil_div.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/PsRoiPoolKernels.h>

namespace at::native::xpu {

template <typename T>
struct PsRoiPoolForwardKernel {
  void operator()(sycl::nd_item<1> item) const {
    XPU_KERNEL_LOOP(item, index, nthreads_) {
      // (n, c_out, ph, pw) is an element in the pooled output
      int pw = index % pooled_width_;
      int ph = (index / pooled_width_) % pooled_height_;
      int c_out = (index / pooled_width_ / pooled_height_) % channels_out_;
      int n = index / pooled_width_ / pooled_height_ / channels_out_;

      // (n, c_in, ph, pw) is the associated element in the input
      int c_in = (c_out * pooled_height_ + ph) * pooled_width_ + pw;

      const T* offset_rois = rois_ + n * 5;
      int roi_batch_ind = offset_rois[0];
      int roi_start_w = std::round(offset_rois[1] * spatial_scale_);
      int roi_start_h = std::round(offset_rois[2] * spatial_scale_);
      int roi_end_w = std::round(offset_rois[3] * spatial_scale_);
      int roi_end_h = std::round(offset_rois[4] * spatial_scale_);

      // Force malformed ROIs to be 1x1
      int roi_width = std::max(roi_end_w - roi_start_w, 1);
      int roi_height = std::max(roi_end_h - roi_start_h, 1);
      T bin_size_h =
          static_cast<T>(roi_height) / static_cast<T>(pooled_height_);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width_);

      int hstart =
          static_cast<int>(std::floor(static_cast<T>(ph) * bin_size_h));
      int wstart =
          static_cast<int>(std::floor(static_cast<T>(pw) * bin_size_w));
      int hend =
          static_cast<int>(std::ceil(static_cast<T>(ph + 1) * bin_size_h));
      int wend =
          static_cast<int>(std::ceil(static_cast<T>(pw + 1) * bin_size_w));

      // Add roi offsets and clip to input boundaries
      hstart = std::min(std::max(hstart + roi_start_h, 0), height_ - 1);
      hend = std::min(std::max(hend + roi_start_h, 0), height_ - 1);
      wstart = std::min(std::max(wstart + roi_start_w, 0), width_ - 1);
      wend = std::min(std::max(wend + roi_start_w, 0), width_ - 1);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      const T* offset_input =
          input_ + (roi_batch_ind * channels_ + c_in) * height_ * width_;
      T out_sum = 0;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int input_index = h * width_ + w;
          out_sum += offset_input[input_index];
        }
      }

      T bin_area = (hend - hstart) * (wend - wstart);
      output_[index] = is_empty ? static_cast<T>(0) : out_sum / bin_area;
      channel_mapping_[index] = c_in;
    }
  }
  PsRoiPoolForwardKernel(
      int nthreads,
      const T* input,
      const T spatial_scale,
      int channels,
      int height,
      int width,
      int pooled_height,
      int pooled_width,
      const T* rois,
      int channels_out,
      T* output,
      int* channel_mapping)
      : nthreads_(nthreads),
        input_(input),
        spatial_scale_(spatial_scale),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        rois_(rois),
        channels_out_(channels_out),
        output_(output),
        channel_mapping_(channel_mapping) {}

 private:
  int nthreads_;
  const T* input_;
  const T spatial_scale_;
  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  const T* rois_;
  int channels_out_;
  T* output_;
  int* channel_mapping_;
};

std::tuple<Tensor, Tensor> ps_roi_pool_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  TORCH_CHECK(
      channels % (pooled_height * pooled_width) == 0,
      "input channels must be a multiple of pooling height * pooling width");
  int channels_out = channels / (pooled_height * pooled_width);

  at::Tensor output = at::zeros(
      {num_rois, channels_out, pooled_height, pooled_width}, input.options());
  at::Tensor channel_mapping =
      at::zeros(output.sizes(), input.options().dtype(at::kInt));

  auto output_size = output.numel();
  int64_t global_range = std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096));
  int64_t local_range = 512;

  if (output_size == 0) {
    return std::make_tuple(output, channel_mapping);
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ps_roi_pool_forward_kernel_xpu", [&] {
        auto kfn = PsRoiPoolForwardKernel<scalar_t>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_.data_ptr<scalar_t>(),
            channels_out,
            output.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>());
        sycl_kernel_submit(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            kfn);
      });
  return std::make_tuple(output, channel_mapping);
}

template <typename T>
struct PsRoiPoolBackwardKernel {
  void operator()(sycl::nd_item<1> item) const {
    XPU_KERNEL_LOOP(item, index, nthreads_) {
      // (n, c, ph, pw) is an element in the pooled output
      int pw = index % pooled_width_;
      int ph = (index / pooled_width_) % pooled_height_;
      int n = index / pooled_width_ / pooled_height_ / channels_out_;

      const T* offset_rois = rois_ + n * 5;
      int roi_batch_ind = offset_rois[0];
      int roi_start_w = std::roundf(offset_rois[1] * spatial_scale_);
      int roi_start_h = std::roundf(offset_rois[2] * spatial_scale_);
      int roi_end_w = std::roundf(offset_rois[3] * spatial_scale_);
      int roi_end_h = std::roundf(offset_rois[4] * spatial_scale_);

      // Force too small ROIs to be 1x1
      int roi_width = std::max(roi_end_w - roi_start_w, 1);
      int roi_height = std::max(roi_end_h - roi_start_h, 1);
      T bin_size_h =
          static_cast<T>(roi_height) / static_cast<T>(pooled_height_);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width_);

      int hstart =
          static_cast<int>(std::floor(static_cast<T>(ph) * bin_size_h));
      int wstart =
          static_cast<int>(std::floor(static_cast<T>(pw) * bin_size_w));
      int hend =
          static_cast<int>(std::ceil(static_cast<T>(ph + 1) * bin_size_h));
      int wend =
          static_cast<int>(std::ceil(static_cast<T>(pw + 1) * bin_size_w));

      // Add roi offsets and clip to input boundaries
      hstart = std::min(std::max(hstart + roi_start_h, 0), height_);
      hend = std::min(std::max(hend + roi_start_h, 0), height_);
      wstart = std::min(std::max(wstart + roi_start_w, 0), width_);
      wend = std::min(std::max(wend + roi_start_w, 0), width_);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      int c_in = channel_mapping_[index];
      T bin_area = (hend - hstart) * (wend - wstart);
      T diff_val =
          is_empty ? static_cast<T>(0) : grad_output_[index] / bin_area;

      const int offset = (roi_batch_ind * channels_ + c_in) * height_ * width_;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int grad_input_index = h * width_ + w;
          atomicAdd(
              (sycl_global_ptr<T>)(grad_input_ + offset + grad_input_index),
              static_cast<T>(diff_val));
        }
      }
    }
  }
  PsRoiPoolBackwardKernel(
      int nthreads,
      const T* grad_output,
      const int* channel_mapping,
      const T spatial_scale,
      int channels,
      int height,
      int width,
      int pooled_height,
      int pooled_width,
      int channels_out,
      T* grad_input,
      const T* rois)
      : nthreads_(nthreads),
        grad_output_(grad_output),
        channel_mapping_(channel_mapping),
        spatial_scale_(spatial_scale),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        channels_out_(channels_out),
        grad_input_(grad_input),
        rois_(rois) {}

 private:
  int nthreads_;
  const T* grad_output_;
  const int* channel_mapping_;
  const T spatial_scale_;
  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  int channels_out_;
  T* grad_input_;
  const T* rois_;
};

Tensor ps_roi_pool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());
  int64_t global_range = std::min(
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096));
  int64_t local_range = 512;

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  int channels_out = channels / (pooled_height * pooled_width);

  auto grad_ = grad.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "ps_roi_pool_backward_kernel_xpu", [&] {
        auto kfn = PsRoiPoolBackwardKernel<scalar_t>(
            grad.numel(),
            grad_.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            channels_out,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>());
        sycl_kernel_submit(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            kfn);
      });
  return grad_input;
}

} // namespace at::native::xpu

#pragma GCC diagnostic pop
#pragma clang diagnostic pop