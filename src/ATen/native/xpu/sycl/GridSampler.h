#pragma once
#include <ATen/native/GridSampler.h>

#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/XPUMathCompat.h>

namespace at::native::xpu {

template <typename scalar_t, typename index_t>
static inline void safe_add_2d(
    scalar_t* data,
    int64_t h,
    int64_t w,
    int64_t sH,
    int64_t sW,
    int64_t H,
    int64_t W,
    scalar_t delta,
    index_t NC_offset) {
  if (within_bounds_2d(h, w, H, W)) {
    atomicAdd(
        (sycl_global_ptr<scalar_t>)&data[NC_offset + h * sH + w * sW], delta);
  }
}

template <typename scalar_t, typename index_t>
static inline void safe_add_3d(
    scalar_t* data,
    int64_t d,
    int64_t h,
    int64_t w,
    int64_t sD,
    int64_t sH,
    int64_t sW,
    int64_t D,
    int64_t H,
    int64_t W,
    scalar_t delta,
    index_t NC_offset) {
  if (within_bounds_3d(d, h, w, D, H, W)) {
    atomicAdd(
        (sycl_global_ptr<scalar_t>)&data[NC_offset + d * sD + h * sH + w * sW],
        delta);
  }
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
  x = at::native::xpu::compute_coordinates(x, W, padding_mode, align_corners);
  y = at::native::xpu::compute_coordinates(y, H, padding_mode, align_corners);

  int64_t ix = static_cast<int64_t>(x);
  int64_t iy = static_cast<int64_t>(y);

  if (within_bounds_2d(iy, ix, H, W)) {
    return data[iy * sH + ix * sW];
  }
  return static_cast<scalar_t>(0);
}

template <typename scalar_t, typename index_t>
static inline void add_value_bounded(
    scalar_t* data,
    scalar_t x,
    scalar_t y,
    int64_t W,
    int64_t H,
    int64_t sW,
    int64_t sH,
    scalar_t delta,
    GridSamplerPadding padding_mode,
    bool align_corners,
    const index_t NC_offset) {
  x = at::native::xpu::compute_coordinates(x, W, padding_mode, align_corners);
  y = at::native::xpu::compute_coordinates(y, H, padding_mode, align_corners);

  int64_t ix = static_cast<int64_t>(x);
  int64_t iy = static_cast<int64_t>(y);

  at::native::xpu::safe_add_2d(data, iy, ix, sH, sW, H, W, delta, NC_offset);
}

template <typename scalar_t>
static inline scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int64_t size,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = at::native::xpu::compute_coordinates(
      coord, size, padding_mode, align_corners);
  return coord;
}

template <typename scalar_t>
static inline scalar_t grid_sampler_compute_source_index_set_grad(
    scalar_t coord,
    int64_t size,
    GridSamplerPadding padding_mode,
    bool align_corners,
    scalar_t* grad_in) {
  scalar_t grad_clip, grad_refl;
  coord =
      grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord =
          reflect_coordinates_set_grad(coord, 0, 2 * (size - 1), &grad_refl);
    } else {
      coord = reflect_coordinates_set_grad(coord, -1, 2 * size - 1, &grad_refl);
    }
    // clip coordinates to image borders
    coord = clip_coordinates_set_grad(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}

} // namespace at::native::xpu
