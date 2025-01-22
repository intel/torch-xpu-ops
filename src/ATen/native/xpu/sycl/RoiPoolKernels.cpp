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

#include <ATen/native/xpu/sycl/RoiPoolKernels.h>

namespace at::native::xpu {

template <typename T>
struct RoiPoolForwardKernel {
  void operator()(sycl::nd_item<1> item) const {
    XPU_KERNEL_LOOP(item, index, nthreads_) {
      int pw = index % pooled_width_;
      int ph = (index / pooled_width_) % pooled_height_;
      int c = (index / pooled_width_ / pooled_height_) % channels_;
      int n = index / pooled_width_ / pooled_height_ / channels_;

      const T* offset_rois = rois_ + n * 5;
      int roi_batch_ind = offset_rois[0];
      int roi_start_w = std::round(offset_rois[1] * spatial_scale_);
      int roi_start_h = std::round(offset_rois[2] * spatial_scale_);
      int roi_end_w = std::round(offset_rois[3] * spatial_scale_);
      int roi_end_h = std::round(offset_rois[4] * spatial_scale_);

      // Force malformed ROIs to be 1x1
      int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
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

      // Define an empty pooling region to be zero
      T maxval = is_empty ? 0.0 : std::numeric_limits<float>::lowest();
      // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
      int maxidx = -1;
      const T* offset_input =
          input_ + (roi_batch_ind * channels_ + c) * height_ * width_;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int input_index = h * width_ + w;
          if (offset_input[input_index] > maxval) {
            maxval = offset_input[input_index];
            maxidx = input_index;
          }
        }
      }
      output_[index] = maxval;
      argmax_[index] = maxidx;
    }
  }
  RoiPoolForwardKernel(
      int nthreads,
      const T* input,
      const T spatial_scale,
      int channels,
      int height,
      int width,
      int pooled_height,
      int pooled_width,
      const T* rois,
      T* output,
      int* argmax)
      : nthreads_(nthreads),
        input_(input),
        spatial_scale_(spatial_scale),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        rois_(rois),
        output_(output),
        argmax_(argmax) {}

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
  T* output_;
  int* argmax_;
};

std::tuple<Tensor, Tensor> roi_pool_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  at::Tensor argmax = at::zeros(
      {num_rois, channels, pooled_height, pooled_width},
      input.options().dtype(at::kInt));

  auto output_size = num_rois * pooled_height * pooled_width * channels;
  int64_t global_range =
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512));
  int64_t local_range = 512;

  if (output.numel() == 0) {
    return std::make_tuple(output, argmax);
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "roi_pool_forward_kernel_xpu", [&] {
        auto kfn = RoiPoolForwardKernel<scalar_t>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int>());
        sycl_kernel_submit(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            kfn);
      });
  return std::make_tuple(output, argmax);
}

template <typename T>
struct RoiPoolBackwardKernel {
  void operator()(sycl::nd_item<1> item) const {
    XPU_KERNEL_LOOP(item, index, nthreads_) {
      // (n, c, ph, pw) is an element in the pooled output
      int pw = index % pooled_width_;
      int ph = (index / pooled_width_) % pooled_height_;
      int c = (index / pooled_width_ / pooled_height_) % channels_;
      int n = index / pooled_width_ / pooled_height_ / channels_;

      const T* offset_rois = rois_ + n * 5;
      int roi_batch_ind = offset_rois[0];

      const int output_offset = n * n_stride_ + c * c_stride_;
      const int* argmax_data_offset =
          argmax_data_ + (n * channels_ + c) * pooled_height_ * pooled_width_;
      const int argmax = argmax_data_offset[ph * pooled_width_ + pw];
      const int offset = (roi_batch_ind * channels_ + c) * height_ * width_;

      if (argmax != -1) {
        atomicAdd(
            (sycl_global_ptr<T>)(grad_input_ + offset + argmax),
            static_cast<T>(
                grad_output_[output_offset + ph * h_stride_ + pw * w_stride_]));
      }
    }
  }
  RoiPoolBackwardKernel(
      int nthreads,
      const T* grad_output,
      const int* argmax_data,
      int num_rois,
      const T spatial_scale,
      int channels,
      int height,
      int width,
      int pooled_height,
      int pooled_width,
      T* grad_input,
      const T* rois,
      int n_stride,
      int c_stride,
      int h_stride,
      int w_stride)
      : nthreads_(nthreads),
        grad_output_(grad_output),
        argmax_data_(argmax_data),
        num_rois_(num_rois),
        spatial_scale_(spatial_scale),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        grad_input_(grad_input),
        rois_(rois),
        n_stride_(n_stride),
        c_stride_(c_stride),
        h_stride_(h_stride),
        w_stride_(w_stride) {}

 private:
  int nthreads_;
  const T* grad_output_;
  const int* argmax_data_;
  int num_rois_;
  const T spatial_scale_;
  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  T* grad_input_;
  const T* rois_;
  int n_stride_;
  int c_stride_;
  int h_stride_;
  int w_stride_;
};

Tensor roi_pool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
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

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  auto num_rois = rois.size(0);
  auto argmax_ = argmax.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "roi_pool_backward_kernel_xpu", [&] {
        auto kfn = RoiPoolBackwardKernel<scalar_t>(
            grad.numel(),
            grad.data_ptr<scalar_t>(),
            argmax_.data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride);
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