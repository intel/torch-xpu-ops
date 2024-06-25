#pragma once
#include <ATen/core/TensorAccessor.h>
#include <ATen/native/xpu/sycl/Atomics.h>

#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <c10/util/OptionalArrayRef.h>
#include <c10/util/SmallVector.h>

#include <math.h>

namespace at::native::xpu {

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

/* TODO: move this to a common place */
template <typename scalar_t>
inline scalar_t min(scalar_t a, scalar_t b) {
  return a < b ? a : b;
}

template <typename scalar_t>
inline scalar_t max(scalar_t a, scalar_t b) {
  return a > b ? a : b;
}

// NOTE [ Nearest neighbor upsampling kernel implementation ]
//
// The nearest neighbor upsampling kernel implementation is symmetrical as
// expected. We launch kernels with threads mapping to destination tensors where
// kernels write data to, each thread reads data from the source tensor, this
// means:
// 1. In the forward kernel,
//      src_xxx refers to properties of input tensors;
//      dst_xxx refers to properties of output tensors;
//      scale_factor is the ratio of src_size to dst_size;
// 2. In the backward kernel,
//      src_xxx refers to properties of grad_output tensors;
//      dst_xxx refers to properties of grad_input tensors;
//      scale_factor is the ratio of src_size to dst_size;
//
// Because of this, we need to take the reciprocal of the scale defined by
// upsample layer during forward path. The motivation is to avoid slow
// division in the kernel code, so we can use faster multiplication instead.
// This is not necessary during backward path, since the scale_factor is already
// the reciprocal of corresponding scale_factor used in the forward path due to
// the swap of source and destination tensor.
//
// Similarly, since the mapping from grad_input to grad_output during backward
// is the reverse of the mapping of output to input, we need to have opposite
// mapping functions to compute the source index.

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename accscalar_t>
static inline accscalar_t compute_scales_value(
    const c10::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  // FIXME: remove magic > 0 after we ensure no models were serialized with -1
  // defaults.
  return (scale.has_value() && scale.value() > 0.)
      ? (accscalar_t)(1.0 / scale.value())
      : (accscalar_t)src_size / dst_size;
}

// see NOTE [ Nearest neighbor upsampling kernel implementation ]
template <typename accscalar_t>
static inline accscalar_t compute_scales_value_backwards(
    const c10::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  // FIXME: remove magic > 0 after we ensure no models were serialized with -1
  // defaults.
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
    // See Note[Follow Opencv resize logic]
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

} // namespace at::native::xpu
