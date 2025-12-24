/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/Resize.h>
#if defined(USE_ONEMKL_XPU)
#include <ATen/native/xpu/mkl/SpectralOps.h>
#else
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
#endif // USE_ONEMKL_XPU

namespace at::native {

namespace {

bool needs_fft_promotion(c10::ScalarType dtype) {
  return dtype == at::kHalf || dtype == at::kComplexHalf;
}

Tensor promote_fft_input(const Tensor& self) {
  const auto dtype = self.scalar_type();
  if (dtype == at::kHalf) {
    return self.to(self.options().dtype(at::kFloat));
  }
  if (dtype == at::kComplexHalf) {
    return self.to(self.options().dtype(at::kComplexFloat));
  }
  return self;
}

Tensor cast_fft_output(const Tensor& result, c10::ScalarType target_dtype) {
  return result.to(result.options().dtype(target_dtype));
}

} // namespace

Tensor _fft_c2c_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_CHECK(self.is_complex());
  const auto input_dtype = self.scalar_type();
  const auto input = promote_fft_input(self);

#if defined(USE_ONEMKL_XPU)
  auto result = native::xpu::_fft_c2c_mkl(input, dim, normalization, forward);
#else
  Tensor out_cpu = native::_fft_c2c_mkl(
      input.to(Device(at::kCPU)), dim, normalization, forward);
  auto result = out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL_XPU
  if (!needs_fft_promotion(input_dtype)) {
    return result;
  }
  return cast_fft_output(result, input_dtype);
}

Tensor& _fft_c2c_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());
  const auto input_dtype = self.scalar_type();
  const auto input = promote_fft_input(self);

#if defined(USE_ONEMKL_XPU)
  if (!needs_fft_promotion(input_dtype)) {
    return native::xpu::_fft_c2c_mkl_out(self, dim, normalization, forward, out);
  }
  auto result = native::xpu::_fft_c2c_mkl(input, dim, normalization, forward);
#else
  Tensor out_cpu = native::_fft_c2c_mkl(
      input.to(Device(at::kCPU)), dim, normalization, forward);
  auto result = out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL_XPU
  if (result.scalar_type() != out.scalar_type()) {
    result = result.to(out.options().dtype(out.scalar_type()));
  }
  at::native::resize_output(out, result.sizes());
  out.copy_(result);
  return out;
}

Tensor _fft_c2r_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size) {
  TORCH_CHECK(self.is_complex());
  const auto input_dtype = self.scalar_type();
  const auto input = promote_fft_input(self);

#if defined(USE_ONEMKL_XPU)
  auto result =
      native::xpu::_fft_c2r_mkl(input, dim, normalization, last_dim_size);
#else
  Tensor out_cpu = native::_fft_c2r_mkl(
      input.to(Device(at::kCPU)), dim, normalization, last_dim_size);
  auto result = out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL_XPU
  if (!needs_fft_promotion(input_dtype)) {
    return result;
  }
  const auto target_dtype = c10::toRealValueType(input_dtype);
  return cast_fft_output(result, target_dtype);
}

Tensor& _fft_c2r_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());
  const auto input_dtype = self.scalar_type();
  const auto input = promote_fft_input(self);

#if defined(USE_ONEMKL_XPU)
  if (!needs_fft_promotion(input_dtype)) {
    return native::xpu::_fft_c2r_mkl_out(
        self, dim, normalization, last_dim_size, out);
  }
  auto result =
      native::xpu::_fft_c2r_mkl(input, dim, normalization, last_dim_size);
#else
  Tensor out_cpu = native::_fft_c2r_mkl(
      input.to(Device(at::kCPU)), dim, normalization, last_dim_size);
  auto result = out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL_XPU
  if (result.scalar_type() != out.scalar_type()) {
    result = result.to(out.options().dtype(out.scalar_type()));
  }
  at::native::resize_output(out, result.sizes());
  out.copy_(result);
  return out;
}

Tensor _fft_r2c_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  const auto input_dtype = self.scalar_type();
  const auto input = promote_fft_input(self);

#if defined(USE_ONEMKL_XPU)
  auto result = native::xpu::_fft_r2c_mkl(input, dim, normalization, onesided);
#else
  Tensor out_cpu = native::_fft_r2c_mkl(
      input.to(Device(at::kCPU)), dim, normalization, onesided);
  auto result = out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL_XPU
  if (!needs_fft_promotion(input_dtype)) {
    return result;
  }
  const auto target_dtype = c10::toComplexType(input_dtype);
  return cast_fft_output(result, target_dtype);
}

Tensor& _fft_r2c_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out) {
  TORCH_CHECK(self.is_floating_point());
  const auto input_dtype = self.scalar_type();
  const auto input = promote_fft_input(self);

#if defined(USE_ONEMKL_XPU)
  if (!needs_fft_promotion(input_dtype)) {
    return native::xpu::_fft_r2c_mkl_out(
        self, dim, normalization, onesided, out);
  }
  auto result = native::xpu::_fft_r2c_mkl(input, dim, normalization, onesided);
#else
  Tensor out_cpu = native::_fft_r2c_mkl(
      input.to(Device(at::kCPU)), dim, normalization, onesided);
  auto result = out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL_XPU
  if (result.scalar_type() != out.scalar_type()) {
    result = result.to(out.options().dtype(out.scalar_type()));
  }
  at::native::resize_output(out, result.sizes());
  out.copy_(result);
  return out;
}

} // namespace at::native
