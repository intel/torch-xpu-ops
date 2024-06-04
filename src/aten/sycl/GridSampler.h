#pragma once
#include <ATen/native/GridSampler.h>
#include <ATen/native/GridSamplerUtils.h>

namespace at::native::xpu {

using namespace at::native;

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