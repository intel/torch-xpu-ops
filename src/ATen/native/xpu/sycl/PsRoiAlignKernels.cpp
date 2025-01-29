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

#include <ATen/native/xpu/sycl/PsRoiAlignKernels.h>

namespace at::native::xpu {

template <typename T>
T bilinear_interpolate(
    const T* input,
    int height,
    int width,
    T y,
    T x,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
struct PsRoiAlignForwardKernel {
  void operator()(sycl::nd_item<1> item) const {
    XPU_KERNEL_LOOP(item, index, nthreads_) {
      // (n, c, ph, pw) is an element in the pooled output
      int pw = index % pooled_width_;
      int ph = (index / pooled_width_) % pooled_height_;
      int c_out = (index / pooled_width_ / pooled_height_) % channels_out_;
      int n = index / pooled_width_ / pooled_height_ / channels_out_;

      // (n, c_in, ph, pw) is the associated element in the input
      int c_in = (c_out * pooled_height_ + ph) * pooled_width_ + pw;

      const T* offset_rois = rois_ + n * 5;
      int roi_batch_ind = offset_rois[0];

      // Do not using rounding; this implementation detail is critical
      T roi_start_w = offset_rois[1] * spatial_scale_ - static_cast<T>(0.5);
      T roi_start_h = offset_rois[2] * spatial_scale_ - static_cast<T>(0.5);
      T roi_end_w = offset_rois[3] * spatial_scale_ - static_cast<T>(0.5);
      T roi_end_h = offset_rois[4] * spatial_scale_ - static_cast<T>(0.5);

      T roi_width = roi_end_w - roi_start_w;
      T roi_height = roi_end_h - roi_start_h;
      T bin_size_h =
          static_cast<T>(roi_height) / static_cast<T>(pooled_height_);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width_);

      // Do not using floor/ceil; this implementation detail is critical
      T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
      T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio_ > 0)
          ? sampling_ratio_
          : std::ceil(roi_height / pooled_height_); // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio_ > 0)
          ? sampling_ratio_
          : std::ceil(roi_width / pooled_width_);
      const T count = roi_bin_grid_h * roi_bin_grid_w;

      const T* offset_input =
          input_ + (roi_batch_ind * channels_ + c_in) * height_ * width_;
      T out_sum = 0.;
      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = hstart +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = wstart +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);
          T val =
              bilinear_interpolate(offset_input, height_, width_, y, x, index);
          out_sum += val;
        }
      }
      out_sum /= count;
      output_[index] = out_sum;
      channel_mapping_[index] = c_in;
    }
  }
  PsRoiAlignForwardKernel(
      int nthreads,
      const T* input,
      const T spatial_scale,
      int channels,
      int height,
      int width,
      int pooled_height,
      int pooled_width,
      int sampling_ratio,
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
        sampling_ratio_(sampling_ratio),
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
  int sampling_ratio_;
  const T* rois_;
  int channels_out_;
  T* output_;
  int* channel_mapping_;
};

template <typename T>
void bilinear_interpolate_gradient(
    int height,
    int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}

template <typename T>
struct PsRoiAlignBackwardKernel {
  void operator()(sycl::nd_item<1> item) const {
    XPU_KERNEL_LOOP(item, index, nthreads_) {
      // (n, c, ph, pw) is an element in the pooled output
      int pw = index % pooled_width_;
      int ph = (index / pooled_width_) % pooled_height_;
      int n = index / pooled_width_ / pooled_height_ / channels_out_;

      const T* offset_rois = rois_ + n * 5;
      int roi_batch_ind = offset_rois[0];

      // Do not using rounding; this implementation detail is critical
      T roi_start_w = offset_rois[1] * spatial_scale_ - static_cast<T>(0.5);
      T roi_start_h = offset_rois[2] * spatial_scale_ - static_cast<T>(0.5);
      T roi_end_w = offset_rois[3] * spatial_scale_ - static_cast<T>(0.5);
      T roi_end_h = offset_rois[4] * spatial_scale_ - static_cast<T>(0.5);

      // Force small ROIs to be 1x1
      T roi_width = roi_end_w - roi_start_w;
      T roi_height = roi_end_h - roi_start_h;
      T bin_size_h =
          static_cast<T>(roi_height) / static_cast<T>(pooled_height_);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width_);

      int c_in = channel_mapping_[index];

      // Do not using floor/ceil; this implementation detail is critical
      T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
      T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;

      const T grad_output_this_bin = grad_output_[index];

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio_ > 0)
          ? sampling_ratio_
          : std::ceil(roi_height / pooled_height_); // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio_ > 0)
          ? sampling_ratio_
          : std::ceil(roi_width / pooled_width_);
      const T count = roi_bin_grid_h * roi_bin_grid_w;

      const int offset = (roi_batch_ind * channels_ + c_in) * height_ * width_;

      for (int iy = 0; iy < roi_bin_grid_h; iy++) {
        const T y = hstart +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h);
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = wstart +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T w1, w2, w3, w4;
          int x_low, x_high, y_low, y_high;

          bilinear_interpolate_gradient(
              height_,
              width_,
              y,
              x,
              w1,
              w2,
              w3,
              w4,
              x_low,
              x_high,
              y_low,
              y_high,
              index);

          T g1 = grad_output_this_bin * w1 / count;
          T g2 = grad_output_this_bin * w2 / count;
          T g3 = grad_output_this_bin * w3 / count;
          T g4 = grad_output_this_bin * w4 / count;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
            atomicAdd(
                (sycl_global_ptr<
                    T>)(grad_input_ + offset + y_low * width_ + x_low),
                static_cast<T>(g1));
            atomicAdd(
                (sycl_global_ptr<
                    T>)(grad_input_ + offset + y_low * width_ + x_high),
                static_cast<T>(g2));
            atomicAdd(
                (sycl_global_ptr<
                    T>)(grad_input_ + offset + y_high * width_ + x_low),
                static_cast<T>(g3));
            atomicAdd(
                (sycl_global_ptr<
                    T>)(grad_input_ + offset + y_high * width_ + x_high),
                static_cast<T>(g4));
          } // if
        } // ix
      } // iy
    } // XPU_KERNEL_LOOP
  }
  PsRoiAlignBackwardKernel(
      int nthreads,
      const T* grad_output,
      const int* channel_mapping,
      const T spatial_scale,
      int channels,
      int height,
      int width,
      int pooled_height,
      int pooled_width,
      int sampling_ratio,
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
        sampling_ratio_(sampling_ratio),
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
  int sampling_ratio_;
  int channels_out_;
  T* grad_input_;
  const T* rois_;
};

std::tuple<at::Tensor, at::Tensor> ps_roi_align_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio) {
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

  if (output.numel() == 0) {
    return std::make_tuple(output, channel_mapping);
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ps_roi_align_forward_kernel_xpu", [&] {
        auto kfn = PsRoiAlignForwardKernel<scalar_t>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
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

Tensor ps_roi_align_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
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

  at::globalContext().alertNotDeterministic("ps_roi_align_backward_kernel_xpu");

  auto grad_ = grad.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "ps_roi_align_backward_kernel_xpu", [&] {
        auto kfn = PsRoiAlignBackwardKernel<scalar_t>(
            grad.numel(),
            grad_.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
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