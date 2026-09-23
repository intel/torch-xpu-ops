/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Portions of this file are derived from PyTorch
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include <ATen/OpMathType.h>
#include <ATen/native/GridSampler.h>

#include <ATen/native/xpu/UpSample.h>
#include <ATen/native/xpu/sycl/Atomics.h>
#include <comm/XPUMathCompat.h>

#include <limits>

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
  // We avoid using double here because some platforms may not support it.
  if (static_cast<int64_t>(x) > INT_MAX - 1 || x < INT_MIN ||
      !sycl::isfinite(static_cast<at::opmath_type<scalar_t>>(x)))
    return static_cast<scalar_t>(-100.0f);
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

// grid_sampler_unnormalize with the extent in index_t, for the kernels that
// index with int64_t. It converts where the int-taking helper converts.
template <typename scalar_t, typename index_t>
static inline scalar_t grid_sampler_unnormalize_sized(
    scalar_t coord,
    index_t size,
    bool align_corners) {
  if (align_corners) {
    return ((coord + 1) / 2) * static_cast<scalar_t>(size - 1);
  } else {
    return ((coord + 1) * static_cast<scalar_t>(size) - 1) / 2;
  }
}

template <typename scalar_t, typename index_t>
static inline scalar_t grid_sampler_unnormalize_set_grad_sized(
    scalar_t coord,
    index_t size,
    bool align_corners,
    scalar_t* grad_in) {
  if (align_corners) {
    *grad_in = static_cast<scalar_t>(size - 1) / 2;
    return ((coord + 1) / 2) * static_cast<scalar_t>(size - 1);
  } else {
    *grad_in = static_cast<scalar_t>(size) / 2;
    return ((coord + 1) * static_cast<scalar_t>(size) - 1) / 2;
  }
}

// compute_coordinates with the extent in index_t, the reflection parity
// taken with fmod and no downgrade: no float converts to an integer, and a
// position past INT_MAX keeps its voxel.
template <typename scalar_t, typename index_t>
static inline scalar_t compute_coordinates_sized(
    scalar_t coord,
    index_t size,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  if (padding_mode == GridSamplerPadding::Border) {
    coord = sycl::fmin(
        static_cast<scalar_t>(size - 1),
        sycl::fmax(coord, static_cast<scalar_t>(0)));
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // the bounds reflect_coordinates halves, formed without doubling the extent
    const scalar_t low =
        align_corners ? static_cast<scalar_t>(0) : static_cast<scalar_t>(-0.5);
    const scalar_t span =
        static_cast<scalar_t>(align_corners ? size - 1 : size);
    if (span == 0) {
      coord = 0;
    } else {
      const scalar_t in = sycl::fabs(coord - low);
      const scalar_t extra = sycl::fmod(in, span);
      const bool odd =
          sycl::fmod(sycl::floor(in / span), static_cast<scalar_t>(2)) != 0;
      coord = odd ? span - extra + low : extra + low;
    }
    coord = sycl::fmin(
        static_cast<scalar_t>(size - 1),
        sycl::fmax(coord, static_cast<scalar_t>(0)));
  }
  return coord;
}

// The four cubic taps one axis contributes at `coord`: the Keys coefficients
// of its fractional part, the index each tap reads, and, when `coeffs_grad` is
// given, the coefficient derivatives. The taps sit around the unclipped index.
// A tap the padding drops takes a negative index, contributes a zero value and
// keeps its coefficient, as get_value_bounded does in 4-D.
template <typename scalar_t, typename index_t>
static inline void resolve_cubic_taps(
    scalar_t coord,
    index_t size,
    GridSamplerPadding padding_mode,
    bool align_corners,
    scalar_t coeffs[4],
    scalar_t* coeffs_grad,
    index_t indices[4]) {
  const scalar_t base = sycl::floor(coord);
  get_cubic_upsampling_coefficients<scalar_t>(coeffs, coord - base);
  if (coeffs_grad != nullptr) {
    get_cubic_coefficients_grad<scalar_t>(coeffs_grad, coord - base);
  }
  const scalar_t index_limit =
      static_cast<scalar_t>(std::numeric_limits<index_t>::max());
#pragma unroll 4
  for (int i = 0; i < 4; ++i) {
    const scalar_t tap = compute_coordinates_sized(
        base - 1 + i, size, padding_mode, align_corners);
    // a tap that is not finite, or past the index type, fails before the cast
    const index_t index = (tap >= 0 && tap < index_limit)
        ? static_cast<index_t>(tap)
        : static_cast<index_t>(-1);
    indices[i] = index < size ? index : static_cast<index_t>(-1);
  }
}

} // namespace at::native::xpu
