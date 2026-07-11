/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// SYCL kernels for _philox_uniform_ and _philox_normal_.
// Ported from CUDA: aten/src/ATen/native/cuda/PhiloxDistribution.cu
// See PyTorch PR #177230.

#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TransformationHelper.h>
#include <ATen/native/xpu/sycl/PhiloxDistributionKernels.h>
#include <ATen/native/xpu/sycl/StatelessPhilox4x32.h>
#include <comm/SYCLContext.h>

#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_normal_native.h>
#include <ATen/ops/_philox_uniform_native.h>
#endif

namespace at::native::xpu {

// Elements produced per Philox 4x32 call:
// 4 for float/half/bfloat16, 2 for double.
template <typename scalar_t>
constexpr int elems_per_call = std::is_same_v<scalar_t, double> ? 2 : 4;

// ─── Uniform transforms ─────────────────────────────────────────

// uint32 → float uniform in [0, 1)
inline float uint32_to_uniform_float(uint32_t val) {
  constexpr uint32_t MASK = static_cast<uint32_t>(
      (static_cast<uint64_t>(1) << std::numeric_limits<float>::digits) - 1);
  constexpr float DIVISOR = static_cast<float>(1.0) /
      static_cast<float>(static_cast<uint32_t>(1)
                         << std::numeric_limits<float>::digits);
  return static_cast<float>(val & MASK) * DIVISOR;
}

// uint64 → double uniform in [0, 1)
inline double uint64_to_uniform_double(uint64_t val) {
  constexpr uint64_t MASK =
      (static_cast<uint64_t>(1) << std::numeric_limits<double>::digits) - 1;
  constexpr double DIVISOR = 1.0 /
      static_cast<double>(static_cast<uint64_t>(1)
                          << std::numeric_limits<double>::digits);
  return static_cast<double>(val & MASK) * DIVISOR;
}

// ─── Box-Muller normal transforms ────────────────────────────────

struct float4 {
  float x, y, z, w;
};
struct double2 {
  double x, y;
};

// Box-Muller: 4 uint32 → 4 standard normal floats
inline float4 box_muller_float(philox_uint4 r) {
  constexpr float M = 2.3283064365386963e-10f; // 1/2^32
  constexpr float TWO_PI = 6.2831853071795864f;
  float u1 = sycl::fma(static_cast<float>(r.x), M, M * 0.5f);
  float u2 = sycl::fma(static_cast<float>(r.y), M, M * 0.5f);
  float u3 = sycl::fma(static_cast<float>(r.z), M, M * 0.5f);
  float u4 = sycl::fma(static_cast<float>(r.w), M, M * 0.5f);

  float radius1 = sycl::sqrt(-2.0f * sycl::log(u1));
  float radius2 = sycl::sqrt(-2.0f * sycl::log(u3));
  float angle1 = TWO_PI * u2;
  float angle2 = TWO_PI * u4;

  return {
      radius1 * sycl::cos(angle1),
      radius1 * sycl::sin(angle1),
      radius2 * sycl::cos(angle2),
      radius2 * sycl::sin(angle2)};
}

// Box-Muller: 4 uint32 → 2 standard normal doubles
inline double2 box_muller_double(philox_uint4 r) {
  constexpr double M = 2.3283064365386963e-10; // 1/2^32
  constexpr double TWO_PI = 6.2831853071795864;
  double u1 = sycl::fma(
      static_cast<double>(r.x),
      M,
      static_cast<double>(r.y) * M * M + M * M * 0.5);
  double u2 = sycl::fma(
      static_cast<double>(r.z),
      M,
      static_cast<double>(r.w) * M * M + M * M * 0.5);
  double radius = sycl::sqrt(-2.0 * sycl::log(u1));
  double angle = TWO_PI * u2;
  return {radius * sycl::cos(angle), radius * sycl::sin(angle)};
}

// ─── Single-key kernel ───────────────────────────────────────────

template <typename scalar_t, bool is_uniform>
struct PhiloxSingleKeyFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int64_t chunk = static_cast<int64_t>(item.get_global_id(0));
    constexpr int epc = elems_per_call<scalar_t>;
    int64_t num_full_chunks = num_elems_ / epc;

    if (chunk < num_full_chunks) {
      auto r = philox_4x32(seed_, offset_ + static_cast<uint64_t>(chunk));
      int64_t base = chunk * epc;
      write_values(r, base, epc);
    }

    // Tail
    if (chunk == num_full_chunks) {
      int64_t tail_start = num_full_chunks * epc;
      int remaining = static_cast<int>(num_elems_ - tail_start);
      if (remaining > 0) {
        auto r = philox_4x32(
            seed_, offset_ + static_cast<uint64_t>(num_full_chunks));
        write_values(r, tail_start, remaining);
      }
    }
  }

  PhiloxSingleKeyFunctor(
      scalar_t* output,
      uint64_t seed,
      uint64_t offset,
      int64_t num_elems,
      scalar_t param0,
      scalar_t param1)
      : output_(output),
        seed_(seed),
        offset_(offset),
        num_elems_(num_elems),
        param0_(param0),
        param1_(param1) {}

 private:
  void write_values(philox_uint4 r, int64_t base, int count) const {
    if constexpr (is_uniform) {
      write_uniform(r, base, count);
    } else {
      write_normal(r, base, count);
    }
  }

  void write_uniform(philox_uint4 r, int64_t base, int count) const {
    if constexpr (std::is_same_v<scalar_t, double>) {
      uint64_t packed[2] = {
          (static_cast<uint64_t>(r.x) << 32) | r.y,
          (static_cast<uint64_t>(r.z) << 32) | r.w};
      for (int j = 0; j < count; j++) {
        double x = uint64_to_uniform_double(packed[j]);
        output_[base + j] = static_cast<scalar_t>(
            x * (static_cast<double>(param1_) - static_cast<double>(param0_)) +
            static_cast<double>(param0_));
      }
    } else {
      uint32_t vals[4] = {r.x, r.y, r.z, r.w};
      for (int j = 0; j < count; j++) {
        float x = uint32_to_uniform_float(vals[j]);
        float result_f =
            x * (static_cast<float>(param1_) - static_cast<float>(param0_)) +
            static_cast<float>(param0_);
        scalar_t val = static_cast<scalar_t>(result_f);
        // For half/bfloat16, rounding can push val to high; step back in
        // the reduced-precision representation.
        if constexpr (
            std::is_same_v<scalar_t, c10::BFloat16> ||
            std::is_same_v<scalar_t, c10::Half>) {
          if (val >= param1_) {
            val.x -= 1;
          }
        }
        output_[base + j] = val;
      }
    }
  }

  void write_normal(philox_uint4 r, int64_t base, int count) const {
    if constexpr (std::is_same_v<scalar_t, double>) {
      auto normals = box_muller_double(r);
      double vals[2] = {normals.x, normals.y};
      for (int j = 0; j < count; j++) {
        output_[base + j] = static_cast<scalar_t>(
            vals[j] * static_cast<double>(param1_) +
            static_cast<double>(param0_));
      }
    } else {
      auto normals = box_muller_float(r);
      float vals[4] = {normals.x, normals.y, normals.z, normals.w};
      for (int j = 0; j < count; j++) {
        output_[base + j] = static_cast<scalar_t>(
            vals[j] * static_cast<float>(param1_) +
            static_cast<float>(param0_));
      }
    }
  }

  scalar_t* output_;
  uint64_t seed_;
  uint64_t offset_;
  int64_t num_elems_;
  scalar_t param0_; // low or mean
  scalar_t param1_; // high or stddev
};

// ─── Distribution dispatch ───────────────────────────────────────

template <bool is_uniform>
void philox_distribution_kernel(
    const char* op_name,
    Tensor& self,
    const Tensor& key,
    double param0,
    double param1) {
  TORCH_CHECK(
      self.is_floating_point(),
      op_name,
      ": self must be a floating point tensor, got ",
      self.scalar_type());
  TORCH_CHECK(
      key.scalar_type() == kUInt64,
      op_name,
      ": key must have dtype uint64, got ",
      key.scalar_type());
  TORCH_CHECK(
      self.device() == key.device(),
      op_name,
      ": self and key must be on the same device, got ",
      self.device(),
      " and ",
      key.device());
  TORCH_CHECK(
      key.dim() >= 1 && key.size(-1) == 2,
      op_name,
      ": key must have shape (2,) or (*batch, 2), got shape ",
      key.sizes());

  if (key.dim() > 1) {
    TORCH_CHECK(
        key.dim() == self.dim() + 1,
        op_name,
        ": batched key must have ndim == output ndim + 1, "
        "got key shape ",
        key.sizes(),
        " with output shape ",
        self.sizes());
    auto key_batch = key.sizes().slice(0, self.dim());
    TORCH_CHECK(
        is_expandable_to(key_batch, self.sizes()),
        op_name,
        ": key batch shape ",
        key_batch,
        " is not broadcastable with output shape ",
        self.sizes());
  }

  if (self.numel() == 0) {
    return;
  }

  // For now, only single-key (non-batched) is implemented.
  // Batched key support can be added following the CUDA multi-key pattern.
  TORCH_CHECK(
      key.dim() == 1,
      op_name,
      ": batched keys not yet supported on XPU, got key shape ",
      key.sizes());
}

template <bool is_uniform>
void philox_distribution_launch(
    Tensor& self,
    const Tensor& key,
    double param0,
    double param1) {
  auto output = self.contiguous();
  auto key_contig = key.contiguous();

  // Key is on device — copy to host to read seed/offset.
  auto key_cpu = key_contig.cpu();
  uint64_t seed = key_cpu.data_ptr<uint64_t>()[0];
  uint64_t offset = key_cpu.data_ptr<uint64_t>()[1];

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      self.scalar_type(),
      is_uniform ? "_philox_uniform_" : "_philox_normal_",
      [&] {
        constexpr int epc = elems_per_call<scalar_t>;
        int64_t num_chunks = (self.numel() + epc - 1) / epc;
        constexpr int block_size = 256;
        int num_blocks =
            static_cast<int>((num_chunks + block_size - 1) / block_size);

        auto functor = PhiloxSingleKeyFunctor<scalar_t, is_uniform>(
            output.mutable_data_ptr<scalar_t>(),
            seed,
            offset,
            self.numel(),
            static_cast<scalar_t>(param0),
            static_cast<scalar_t>(param1));

        sycl_kernel_submit(
            sycl::range<1>(num_blocks * block_size),
            sycl::range<1>(block_size),
            at::xpu::getCurrentSYCLQueue(),
            functor);

        if (output.data_ptr() != self.data_ptr()) {
          self.copy_(output);
        }
      });
}

Tensor& _philox_uniform_xpu_(
    Tensor& self,
    const Tensor& key,
    double low,
    double high) {
  philox_distribution_kernel</*is_uniform=*/true>(
      "_philox_uniform_", self, key, low, high);
  if (self.numel() > 0) {
    philox_distribution_launch</*is_uniform=*/true>(self, key, low, high);
  }
  return self;
}

Tensor& _philox_normal_xpu_(
    Tensor& self,
    const Tensor& key,
    double mean,
    double stddev) {
  philox_distribution_kernel</*is_uniform=*/false>(
      "_philox_normal_", self, key, mean, stddev);
  if (self.numel() > 0) {
    philox_distribution_launch</*is_uniform=*/false>(self, key, mean, stddev);
  }
  return self;
}

} // namespace at::native::xpu
