/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

namespace at {
namespace native {
namespace xpu {

void _fft_fill_with_conjugate_symmetry_xpu(
    ScalarType dtype,
    IntArrayRef mirror_dims,
    IntArrayRef signal_half_sizes,
    IntArrayRef in_strides,
    const void* in_data,
    IntArrayRef out_strides,
    void* out_data);

template <typename scalar_t, typename inp_calc_t, typename out_calc_t>
void _fft_conjugate_copy_kernel(
    int64_t numel,
    scalar_t* out_data,
    const scalar_t* in_data,
    inp_calc_t ic,
    out_calc_t oc);

void _fft_fill_with_conjugate_symmetry_(const Tensor& input, IntArrayRef dim_);

} // namespace xpu
} // namespace native
} // namespace at
