#pragma once
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/xpu/sycl/Atomics.h>

#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <c10/util/SmallVector.h>

#include <ATen/TensorUtils.h>
#include <math.h>

namespace at::native::xpu {

[[maybe_unused]] inline std::array<int64_t, 4> upsample_2d_common_check(
    IntArrayRef input_size,
    IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 2,
      "It is expected output_size equals to 2, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 4,
      "It is expected input_size equals to 4, but got size ",
      input_size.size());

  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_height = input_size[2];
  int64_t input_width = input_size[3];

  TORCH_CHECK(
      input_height > 0 && input_width > 0 && output_height > 0 &&
          output_width > 0,
      "Input and output sizes should be greater than 0,"
      " but got input (H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  return {nbatch, channels, output_height, output_width};
}

[[maybe_unused]] inline std::array<int64_t, 5> upsample_3d_common_check(
    IntArrayRef input_size,
    IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 3,
      "It is expected output_size equals to 3, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 5,
      "It is expected input_size equals to 5, but got size ",
      input_size.size());

  int64_t output_depth = output_size[0];
  int64_t output_height = output_size[1];
  int64_t output_width = output_size[2];

  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_depth = input_size[2];
  int64_t input_height = input_size[3];
  int64_t input_width = input_size[4];

  TORCH_CHECK(
      input_depth > 0 && input_height > 0 && input_width > 0 &&
          output_depth > 0 && output_height > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (D: ",
      input_depth,
      ", H: ",
      input_height,
      ", W: ",
      input_width,
      ") output (D: ",
      output_depth,
      ", H: ",
      output_height,
      ", W: ",
      output_width,
      ")");

  return {nbatch, channels, output_depth, output_height, output_width};
}
inline size_t idx_cl(
    const size_t n,
    const size_t h,
    const size_t w,
    const size_t c,
    const size_t height,
    const size_t width,
    const size_t channel) {
  return ((n * height + h) * width + w) * channel + c;
}

template <typename scalar_t>
inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

template <typename accscalar_t>
static inline accscalar_t compute_scales_value(
    const c10::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  return (scale.has_value() && scale.value() > 0.)
      ? (accscalar_t)(1.0 / scale.value())
      : (accscalar_t)src_size / dst_size;
}

template <typename accscalar_t>
static inline accscalar_t compute_scales_value_backwards(
    const c10::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  return (scale.has_value() && scale.value() > 0.)
      ? (accscalar_t)scale.value()
      : (accscalar_t)src_size / dst_size;
}

template <typename accscalar_t>
static inline accscalar_t area_pixel_compute_scale(
    int input_size,
    int output_size,
    bool align_corners,
    const c10::optional<double> scale) {
  if (align_corners) {
    if (output_size > 1) {
      return (accscalar_t)(input_size - 1) / (output_size - 1);
    } else {
      return static_cast<accscalar_t>(0);
    }
  } else {
    return compute_scales_value<accscalar_t>(scale, input_size, output_size);
  }
}

template <typename accscalar_t>
static inline accscalar_t area_pixel_compute_source_index(
    accscalar_t scale,
    int dst_index,
    bool align_corners,
    bool cubic) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    accscalar_t src_idx = scale * (dst_index + static_cast<accscalar_t>(0.5)) -
        static_cast<accscalar_t>(0.5);

    return (!cubic && src_idx < static_cast<accscalar_t>(0))
        ? static_cast<accscalar_t>(0)
        : src_idx;
  }
}

struct NearestIndexOp {
  int operator()(const float scale, int dst_index, int input_size) const {
    const int src_index =
        min(static_cast<int>(floorf((dst_index)*scale)), input_size - 1);
    return src_index;
  }
};

struct NearestExactIndexOp {
  int operator()(const float scale, int dst_index, int input_size) const {
    const int src_index = min(
        static_cast<int>(floorf((dst_index + static_cast<float>(0.5)) * scale)),
        input_size - 1);
    return src_index;
  }
};

struct NearestBwIndexOp {
  int operator()(const float scale, int dst_index, int output_size) const {
    const int src_index =
        min(static_cast<int>(ceilf(dst_index * scale)), output_size);
    return src_index;
  }
};

struct NearestExactBwIndexOp {
  int operator()(const float scale, int dst_index, int output_size) const {
    const int src_index = min(
        static_cast<int>(ceilf(dst_index * scale - static_cast<float>(0.5))),
        output_size);
    return src_index;
  }
};

template <typename scalar_t>
static inline scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

template <typename scalar_t>
static inline scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

template <typename scalar_t>
static inline void get_cubic_upsample_coefficients(
    scalar_t coeffs[4],
    scalar_t t) {
  scalar_t A = -0.75f;

  scalar_t x1 = t;
  coeffs[0] = cubic_convolution2<scalar_t>(x1 + 1.0f, A);
  coeffs[1] = cubic_convolution1<scalar_t>(x1, A);

  scalar_t x2 = 1.0f - t;
  coeffs[2] = cubic_convolution1<scalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<scalar_t>(x2 + 1.0f, A);
}

template <typename accscalar_t>
static inline void get_cubic_upsampling_coefficients(
    accscalar_t coeffs[4],
    accscalar_t t) {
  accscalar_t A = -0.75;

  accscalar_t x1 = t;
  coeffs[0] = cubic_convolution2<accscalar_t>(x1 + 1.0, A);
  coeffs[1] = cubic_convolution1<accscalar_t>(x1, A);

  // opposite coefficients
  accscalar_t x2 = 1.0 - t;
  coeffs[2] = cubic_convolution1<accscalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<accscalar_t>(x2 + 1.0, A);
}

template <typename scalar_t, typename accscalar_t>
static inline accscalar_t cubic_interp1d(
    scalar_t x0,
    scalar_t x1,
    scalar_t x2,
    scalar_t x3,
    accscalar_t t) {
  accscalar_t coeffs[4];
  get_cubic_upsampling_coefficients<accscalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

template <typename scalar_t>
static scalar_t upsample_get_value_bounded(
    const PackedTensorAccessor64<const scalar_t, 4>& data,
    int batch,
    int channel,
    int width,
    int height,
    int x,
    int y) {
  int access_x = max(min(x, width - 1), static_cast<int>(0));
  int access_y = max(min(y, height - 1), static_cast<int>(0));
  return data[batch][channel][access_y][access_x];
}

template <typename scalar_t, typename accscalar_t>
static void upsample_increment_value_bounded(
    PackedTensorAccessor64<scalar_t, 4>& data,
    int batch,
    int channel,
    int height,
    int width,
    int y,
    int x,
    accscalar_t value) {
  int access_y = max(min(y, height - 1), 0);
  int access_x = max(min(x, width - 1), 0);
  atomicAdd(
      (sycl_global_ptr<scalar_t>)(&data[batch][channel][access_y][access_x]),
      static_cast<scalar_t>(value));
}

[[maybe_unused]] inline std::array<int64_t, 3> upsample_1d_common_check(
    IntArrayRef input_size,
    IntArrayRef output_size) {
  TORCH_CHECK(
      output_size.size() == 1,
      "It is expected output_size equals to 1, but got size ",
      output_size.size());

  TORCH_CHECK(
      input_size.size() == 3,
      "It is expected input_size equals to 3, but got size ",
      input_size.size());

  int64_t output_width = output_size[0];
  int64_t nbatch = input_size[0];
  int64_t channels = input_size[1];
  int64_t input_width = input_size[2];

  TORCH_CHECK(
      input_width > 0 && output_width > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      input_width,
      ") and output (W: ",
      output_width,
      ")");

  return {nbatch, channels, output_width};
}

namespace upsample_antialias {

// taken from
// https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
// src/libImaging/Resample.c#L20-L29
struct BilinearFilterFunctor {
  template <typename accscalar_t>
  accscalar_t operator()(accscalar_t x) const {
    if (x < 0) {
      x = -x;
    }
    if (x < 1) {
      return 1 - x;
    }
    return 0;
  }

  static const int size = 2;
};

// taken from
// https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
// src/libImaging/Resample.c#L46-L62
struct BicubicFilterFunctor {
  template <typename accscalar_t>
  accscalar_t operator()(accscalar_t x) const {
    // https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
    const accscalar_t a = -0.5;
    if (x < 0) {
      x = -x;
    }
    if (x < 1) {
      return ((a + 2) * x - (a + 3)) * x * x + 1;
    }
    if (x < 2) {
      return (((x - 5) * x + 8) * x - 4) * a;
    }
    return 0;
  }

  static const int size = 4;
};

template <typename accscalar_t>
static inline void _compute_weights_span(
    const int i,
    const int input_size,
    const accscalar_t scale,
    const accscalar_t support,
    int& xmin,
    int& xsize,
    accscalar_t& center) {
  center = scale * (i + static_cast<accscalar_t>(0.5));
  xmin =
      max(static_cast<int>(center - support + static_cast<accscalar_t>(0.5)),
          static_cast<int>(0));
  xsize =
      min(static_cast<int>(center + support + static_cast<accscalar_t>(0.5)),
          input_size) -
      xmin;
}

template <typename scalar_t, typename accscalar_t, typename interp_filter_t>
static inline void _compute_weights(
    scalar_t* wt_ptr,
    const accscalar_t scale,
    int interp_size,
    const interp_filter_t& interp_filter,
    accscalar_t xmin_m_center,
    int xsize) {
  accscalar_t invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
  accscalar_t total_w = 0.0;
  int j = 0;
  for (j = 0; j < xsize; j++) {
    accscalar_t w = interp_filter(
        (j + xmin_m_center + static_cast<accscalar_t>(0.5)) * invscale);
    wt_ptr[j] = static_cast<scalar_t>(w);
    total_w += w;
  }
  for (j = 0; j < xsize; j++) {
    if (total_w != 0.0) {
      wt_ptr[j] /= total_w;
    }
  }
  for (; j < interp_size; j++) {
    wt_ptr[j] = static_cast<scalar_t>(0.0);
  }
}

template <typename scalar_t, typename accscalar_t>
static inline accscalar_t interpolate_aa_single_dim(
    const scalar_t* src,
    const scalar_t* weights,
    int size) {
  scalar_t t = static_cast<accscalar_t>(*src);
  scalar_t wts = static_cast<accscalar_t>(weights[0]);
  accscalar_t output = t * wts;

  int j = 1;
  for (; j < size; j++) {
    wts = static_cast<accscalar_t>(weights[j]);
    t = static_cast<accscalar_t>(*(src + j));
    output += t * wts;
  }
  return output;
}

} // namespace upsample_antialias

} // namespace at::native::xpu
