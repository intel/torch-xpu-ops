/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#if defined(USE_ONEMKL_XPU)
#include <ATen/native/xpu/mkl/SpectralOps.h>
#else
#include <ATen/native/Resize.h>
#include <ATen/ops/_fft_c2c_native.h>
#include <ATen/ops/_fft_c2r_native.h>
#include <ATen/ops/_fft_r2c_native.h>
#endif // USE_ONEMKL_XPU

namespace at::native {

#if !defined(USE_ONEMKL_XPU)
namespace {

Tensor promote_fft_input_for_cpu_mkl(const Tensor& input) {
  if (input.scalar_type() == ScalarType::Half) {
    return input.to(ScalarType::Float);
  }
  if (input.scalar_type() == ScalarType::ComplexHalf) {
    return input.to(ScalarType::ComplexFloat);
  }
  return input;
}

} // namespace
#endif

Tensor _fft_c2c_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward) {
  TORCH_CHECK(self.is_complex());
  const auto expected_dtype = self.scalar_type();

  Tensor result;

#if defined(USE_ONEMKL_XPU)
  result = native::xpu::_fft_c2c_mkl(self, dim, normalization, forward);
#else
  Tensor promoted_self = promote_fft_input_for_cpu_mkl(self);
  Tensor out_cpu = native::_fft_c2c_mkl(
      promoted_self.to(Device(at::kCPU)), dim, normalization, forward);
  if (self.scalar_type() == ScalarType::ComplexHalf) {
    out_cpu = out_cpu.to(ScalarType::ComplexHalf);
  }
  result = out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL_XPU

  if (result.scalar_type() != expected_dtype) {
    result = result.to(expected_dtype);
  }
  return result;
}

Tensor& _fft_c2c_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool forward,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());

#if defined(USE_ONEMKL_XPU)
  return native::xpu::_fft_c2c_mkl_out(self, dim, normalization, forward, out);
#else
  Tensor promoted_self = promote_fft_input_for_cpu_mkl(self);
  Tensor out_cpu = native::_fft_c2c_mkl(
      promoted_self.to(Device(at::kCPU)), dim, normalization, forward);
  if (self.scalar_type() == ScalarType::ComplexHalf) {
    out_cpu = out_cpu.to(ScalarType::ComplexHalf);
  }
  at::native::resize_output(out, out_cpu.sizes());
  out.copy_(out_cpu);
  return out;
#endif // USE_ONEMKL_XPU
}

Tensor _fft_c2r_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size) {
  TORCH_CHECK(self.is_complex());
  const auto expected_dtype = c10::toRealValueType(self.scalar_type());

  Tensor result;

#if defined(USE_ONEMKL_XPU)
  result = native::xpu::_fft_c2r_mkl(self, dim, normalization, last_dim_size);
#else
  Tensor promoted_self = promote_fft_input_for_cpu_mkl(self);
  Tensor out_cpu = native::_fft_c2r_mkl(
      promoted_self.to(Device(at::kCPU)), dim, normalization, last_dim_size);
  if (self.scalar_type() == ScalarType::ComplexHalf) {
    out_cpu = out_cpu.to(ScalarType::Half);
  }
  result = out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL_XPU

  if (result.scalar_type() != expected_dtype) {
    result = result.to(expected_dtype);
  }
  return result;
}

Tensor& _fft_c2r_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    int64_t last_dim_size,
    Tensor& out) {
  TORCH_CHECK(self.is_complex());

#if defined(USE_ONEMKL_XPU)
  return native::xpu::_fft_c2r_mkl_out(
      self, dim, normalization, last_dim_size, out);
#else
  Tensor promoted_self = promote_fft_input_for_cpu_mkl(self);
  Tensor out_cpu = native::_fft_c2r_mkl(
      promoted_self.to(Device(at::kCPU)), dim, normalization, last_dim_size);
  if (self.scalar_type() == ScalarType::ComplexHalf) {
    out_cpu = out_cpu.to(ScalarType::Half);
  }
  at::native::resize_output(out, out_cpu.sizes());
  out.copy_(out_cpu);
  return out;
#endif // USE_ONEMKL_XPU
}

Tensor _fft_r2c_xpu(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  TORCH_CHECK(self.is_floating_point());
  const auto expected_dtype = c10::toComplexType(self.scalar_type());

  Tensor result;

#if defined(USE_ONEMKL_XPU)
  result = native::xpu::_fft_r2c_mkl(self, dim, normalization, onesided);
#else
  Tensor promoted_self = promote_fft_input_for_cpu_mkl(self);
  Tensor out_cpu = native::_fft_r2c_mkl(
      promoted_self.to(Device(at::kCPU)), dim, normalization, onesided);
  if (self.scalar_type() == ScalarType::Half) {
    out_cpu = out_cpu.to(ScalarType::ComplexHalf);
  }
  result = out_cpu.to(Device(at::kXPU));
#endif // USE_ONEMKL_XPU

  if (result.scalar_type() != expected_dtype) {
    result = result.to(expected_dtype);
  }
  return result;
}

Tensor& _fft_r2c_xpu_out(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided,
    Tensor& out) {
  TORCH_CHECK(self.is_floating_point());

#if defined(USE_ONEMKL_XPU)
  return native::xpu::_fft_r2c_mkl_out(self, dim, normalization, onesided, out);
#else
  Tensor promoted_self = promote_fft_input_for_cpu_mkl(self);
  Tensor out_cpu = native::_fft_r2c_mkl(
      promoted_self.to(Device(at::kCPU)), dim, normalization, onesided);
  if (self.scalar_type() == ScalarType::Half) {
    out_cpu = out_cpu.to(ScalarType::ComplexHalf);
  }
  at::native::resize_output(out, out_cpu.sizes());
  out.copy_(out_cpu);
  return out;
#endif // USE_ONEMKL_XPU
}

} // namespace at::native
