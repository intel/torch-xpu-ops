#pragma once
#include <ATen/native/GridSamplerUtils.h>
#include <comm/XPUMathCompat.h>
namespace at::native::xpu {

static inline bool within_bounds_2d(
    int64_t h,
    int64_t w,
    int64_t H,
    int64_t W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename scalar_t>
static inline scalar_t safe_downgrade_to_int_range(scalar_t x) {
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior.
  if (static_cast<int64_t>(x) > INT_MAX - 1 || x < INT_MIN ||
      !std::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

template <typename scalar_t>
static inline scalar_t reflect_coordinates(
    scalar_t in,
    int64_t twice_low,
    int64_t twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = std::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = std::fmod(in, span);
  int flips = static_cast<int>(c10::xpu::compat::div_trunc(in, span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template <typename scalar_t>
static inline scalar_t clip_coordinates(scalar_t in, int64_t clip_limit) {
  return std::min(
      static_cast<scalar_t>(clip_limit - 1),
      std::max(in, static_cast<scalar_t>(0)));
}

template <typename scalar_t>
static inline scalar_t compute_coordinates(
    scalar_t coord,
    int size,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

template <typename scalar_t>
static inline scalar_t get_value_bounded(
    const scalar_t* data,
    scalar_t x,
    scalar_t y,
    int64_t W,
    int64_t H,
    int64_t sW,
    int64_t sH,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);

  int64_t ix = static_cast<int64_t>(x);
  int64_t iy = static_cast<int64_t>(y);

  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return static_cast<scalar_t>(0);
}

template <typename scalar_t>
static inline scalar_t grid_sampler_unnormalize(
    scalar_t coord,
    int64_t size,
    bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1) * size - 1) / 2;
  }
}

template <typename scalar_t>
static inline scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int64_t size,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;
}

} // namespace at::native::xpu