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

#include <ATen/native/xpu/sycl/RoiAlignKernels.h>

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

  y = std::max(T(0), y);
  x = std::max(T(0), x);

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  y_low = std::min(height - 1, y_low);
  x_low = std::min(width - 1, x_low);
  y_high = std::min(y_low + 1, height - 1);
  x_high = std::min(x_low + 1, width - 1);

  if (y_low == height - 1) {
    y = (T)y_low;
  }

  if (x_low == width - 1) {
    x = (T)x_low;
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
struct RoiAlignForwardKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<1> item) const {
    // each roi will have 5 values, batch_idx,x1,y1,x2,y2
    constexpr int roi_size = 5;
    auto wg = item.get_group(0);
    int n = wg / wgs_per_roi_;
    int output_index_on_batch_n =
        (wg - n * wgs_per_roi_) * item.get_local_range(0) +
        item.get_local_id(0);
    const T* current_roi = rois_ + n * roi_size;
    if (item.get_local_id(0) == 0) {
      cached_roi_[0] = current_roi[0];

      // Do not using rounding; this implementation detail is critical
      T offset = aligned_ ? (T)0.5 : (T)0.0;
      cached_roi_[1] = current_roi[1] * spatial_scale_ - offset;
      cached_roi_[2] = current_roi[2] * spatial_scale_ - offset;
      cached_roi_[3] = current_roi[3] * spatial_scale_ - offset;
      cached_roi_[4] = current_roi[4] * spatial_scale_ - offset;
    }
    item.barrier(sycl_local_fence);

    if (output_index_on_batch_n < items_per_roi_) {
      int pw = output_index_on_batch_n % pooled_width_;
      int ph = (output_index_on_batch_n / pooled_width_) % pooled_height_;
      int c = (output_index_on_batch_n / pooled_width_ / pooled_height_) %
          channels_;

      int roi_batch_ind = cached_roi_[0];
      T roi_start_w = cached_roi_[1];
      T roi_start_h = cached_roi_[2];
      T roi_end_w = cached_roi_[3];
      T roi_end_h = cached_roi_[4];

      T roi_width = roi_end_w - roi_start_w;
      T roi_height = roi_end_h - roi_start_h;
      if (!aligned_) {
        // Force malformed ROIs to be 1x1
        roi_width = std::max(roi_width, (T)1.);
        roi_height = std::max(roi_height, (T)1.);
      }

      T bin_size_h =
          static_cast<T>(roi_height) / static_cast<T>(pooled_height_);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width_);

      const T* offset_input =
          input_ + (roi_batch_ind * channels_ + c) * height_ * width_;

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio_ > 0)
          ? sampling_ratio_
          : std::ceil(roi_height / pooled_height_); // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio_ > 0)
          ? sampling_ratio_
          : std::ceil(roi_width / pooled_width_);

      // We do average (integral) pooling inside a bin
      // When the grid is empty, output zeros.
      const T count = std::max(
          (int)(roi_bin_grid_h * roi_bin_grid_w), (int)(1)); // e.g. = 4

      T output_val = 0.;
      for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
      {
        const T y = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T val = bilinear_interpolate(
              offset_input,
              height_,
              width_,
              y,
              x,
              output_index_on_batch_n + n * items_per_roi_);
          output_val += val;
        }
      }
      output_val /= count;

      output_[output_index_on_batch_n + n * items_per_roi_] = output_val;
    }
  }
  RoiAlignForwardKernel(
      const T* input,
      const T spatial_scale,
      int items_per_rois,
      int wgs_per_roi,
      int channels,
      int height,
      int width,
      int pooled_height,
      int pooled_width,
      int sampling_ratio,
      bool aligned,
      const T* rois,
      T* output)
      : input_(input),
        spatial_scale_(spatial_scale),
        items_per_roi_(items_per_rois),
        wgs_per_roi_(wgs_per_roi),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        sampling_ratio_(sampling_ratio),
        aligned_(aligned),
        rois_(rois),
        output_(output) {}
  void sycl_ker_config_convention(sycl::handler& cgh) {
    // each roi will have 5 values, batch_idx,x1,y1,x2,y2
    cached_roi_ = sycl_local_acc_t<T>(5, cgh);
  }

 private:
  const T* input_;
  const T spatial_scale_;
  const int items_per_roi_;
  const int wgs_per_roi_;
  const int channels_;
  const int height_;
  const int width_;
  const int pooled_height_;
  const int pooled_width_;
  const int sampling_ratio_;
  const bool aligned_;
  const T* rois_;
  T* output_;
  sycl_local_acc_t<T> cached_roi_;
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
struct RoiAlignBackwardKernel {
  void operator()(sycl::nd_item<1> item) const {
    XPU_KERNEL_LOOP(item, index, nthreads_) {
      // (n, c, ph, pw) is an element in the pooled output
      int pw = index % pooled_width_;
      int ph = (index / pooled_width_) % pooled_height_;
      int c = (index / pooled_width_ / pooled_height_) % channels_;
      int n = index / pooled_width_ / pooled_height_ / channels_;

      const T* offset_rois = rois_ + n * 5;
      int roi_batch_ind = offset_rois[0];

      // Do not using rounding; this implementation detail is critical
      T offset = aligned_ ? (T)0.5 : (T)0.0;
      T roi_start_w = offset_rois[1] * spatial_scale_ - offset;
      T roi_start_h = offset_rois[2] * spatial_scale_ - offset;
      T roi_end_w = offset_rois[3] * spatial_scale_ - offset;
      T roi_end_h = offset_rois[4] * spatial_scale_ - offset;

      T roi_width = roi_end_w - roi_start_w;
      T roi_height = roi_end_h - roi_start_h;
      if (!aligned_) {
        // Force malformed ROIs to be 1x1
        roi_width = std::max(roi_width, (T)1.);
        roi_height = std::max(roi_height, (T)1.);
      }

      T bin_size_h =
          static_cast<T>(roi_height) / static_cast<T>(pooled_height_);
      T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width_);

      // We need to index the gradient using the tensor strides to access the
      // correct values.
      const int output_offset = n * n_stride_ + c * c_stride_;
      const T* offset_grad_output = grad_output_ + output_offset;
      const T grad_output_this_bin =
          offset_grad_output[ph * h_stride_ + pw * w_stride_];

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio_ > 0)
          ? sampling_ratio_
          : std::ceil(roi_height / pooled_height_); // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio_ > 0)
          ? sampling_ratio_
          : std::ceil(roi_width / pooled_width_);

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

      const int input_offset =
          (roi_batch_ind * channels_ + c) * height_ * width_;

      for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
      {
        const T y = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
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
                    T>)(grad_input_ + input_offset + y_low * width_ + x_low),
                static_cast<T>(g1));

            atomicAdd(
                (sycl_global_ptr<
                    T>)(grad_input_ + input_offset + y_low * width_ + x_high),
                static_cast<T>(g2));
            atomicAdd(
                (sycl_global_ptr<
                    T>)(grad_input_ + input_offset + y_high * width_ + x_low),
                static_cast<T>(g3));
            atomicAdd(
                (sycl_global_ptr<
                    T>)(grad_input_ + input_offset + y_high * width_ + x_high),
                static_cast<T>(g4));
          } // if
        } // ix
      } // iy
    } // XPU_KERNEL_LOOP
  }
  RoiAlignBackwardKernel(
      int nthreads,
      const T* grad_output,
      const T spatial_scale,
      int channels,
      int height,
      int width,
      int pooled_height,
      int pooled_width,
      int sampling_ratio,
      bool aligned,
      T* grad_input,
      const T* rois,
      int n_stride,
      int c_stride,
      int h_stride,
      int w_stride,
      const int memory_span)
      : nthreads_(nthreads),
        grad_output_(grad_output),
        spatial_scale_(spatial_scale),
        channels_(channels),
        height_(height),
        width_(width),
        pooled_height_(pooled_height),
        pooled_width_(pooled_width),
        sampling_ratio_(sampling_ratio),
        aligned_(aligned),
        grad_input_(grad_input),
        rois_(rois),
        n_stride_(n_stride),
        c_stride_(c_stride),
        h_stride_(h_stride),
        w_stride_(w_stride),
        memory_span_(memory_span) {}

 private:
  int nthreads_;
  const T* grad_output_;
  const T spatial_scale_;
  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
  bool aligned_;
  T* grad_input_;
  const T* rois_;
  int n_stride_;
  int c_stride_;
  int h_stride_;
  int w_stride_;
  const int memory_span_;
};

Tensor roi_align_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;

  if (output.numel() == 0) {
    return output;
  }

  auto input_ = input.contiguous();
  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "roi_align_forward_kernel_xpu",
      [&] {
        int64_t local_range =
            syclMaxWorkGroupSize<RoiAlignForwardKernel<scalar_t>>();
        int items_per_roi = pooled_height * pooled_width * channels;
        if (items_per_roi < local_range) {
          constexpr int simd_len = 32;
          local_range = std::min(
              local_range,
              int64_t(items_per_roi + simd_len - 1) / simd_len * simd_len);
        }
        int wgs_per_roi = (items_per_roi + local_range - 1) / local_range;
        int64_t global_range = wgs_per_roi * num_rois;
        auto kfn = RoiAlignForwardKernel<scalar_t>(
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            items_per_roi,
            wgs_per_roi,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
        sycl_kernel_submit(
            global_range * local_range,
            local_range,
            at::xpu::getCurrentSYCLQueue(),
            kfn);
      });
  return output;
}

Tensor roi_align_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());
  int64_t global_range =
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512));
  int64_t local_range = 512;

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  at::globalContext().alertNotDeterministic("roi_align_backward_kernel_xpu");

  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      grad.scalar_type(),
      "roi_align_backward_kernel_xpu",
      [&] {
        auto kfn = RoiAlignBackwardKernel<scalar_t>(
            grad.numel(),
            grad.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride,
            grad_input.numel());
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
