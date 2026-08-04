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
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TransformationHelper.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/native/xpu/sycl/PhiloxDistributionKernels.h>
#include <comm/DeviceProperties.h>
#include <comm/SYCLContext.h>

#include <algorithm>
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

// Stateless Philox-4x32: 4 pseudo-random uint32 determined entirely by
// (seed, offset). The subsequence half of the counter is fixed to 0 so the
// whole counter space is addressed by the 64-bit offset alone, which keeps
// the generated values consistent across devices.
inline uint4 philox_4x32(uint64_t seed, uint64_t offset) {
  const uint2 key = {
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  const uint4 counter = {
      static_cast<uint32_t>(offset), static_cast<uint32_t>(offset >> 32), 0, 0};
  return philox4x32_10(counter, key);
}

// --- Box-Muller normal transforms ---

// Box-Muller: 4 uint32 -> 4 standard normal floats
inline float4 box_muller_float(uint4 r) {
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

// Box-Muller: 4 uint32 -> 2 standard normal doubles
inline double2 box_muller_double(uint4 r) {
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

// --- Single-key kernel ---

template <typename scalar_t, bool is_uniform>
struct PhiloxSingleKeyFunctor {
  // Uniform masks the raw bits against the output dtype's mantissa, so its
  // bounds stay in scalar_t. Normal is transformed in compute precision.
  using param_t = std::conditional_t<
      is_uniform,
      scalar_t,
      std::conditional_t<std::is_same_v<scalar_t, double>, double, float>>;

  void operator()(sycl::nd_item<1> item) const {
    auto key_vec = memory::ld_vec<16>(key_);
    auto* key_vals = reinterpret_cast<const uint64_t*>(&key_vec);
    uint64_t seed = key_vals[0];
    uint64_t offset = key_vals[1];

    constexpr int epc = elems_per_call<scalar_t>;
    int64_t num_full_chunks = num_elems_ / epc;
    int64_t num_chunks = (num_elems_ + epc - 1) / epc;

    XPU_KERNEL_LOOP_TYPE(item, chunk, num_chunks, int64_t) {
      int64_t base = chunk * epc;
      // The last chunk is partial when num_elems_ is not a multiple of epc.
      int count =
          chunk < num_full_chunks ? epc : static_cast<int>(num_elems_ - base);
      auto r = philox_4x32(seed, offset + static_cast<uint64_t>(chunk));
      write_values(r, base, count);
    }
  }

  PhiloxSingleKeyFunctor(
      scalar_t* output,
      const uint64_t* key,
      int64_t num_elems,
      param_t param0,
      param_t param1)
      : output_(output),
        key_(key),
        num_elems_(num_elems),
        param0_(param0),
        param1_(param1) {}

 private:
  void write_values(uint4 r, int64_t base, int count) const {
    if constexpr (is_uniform) {
      write_uniform(r, base, count);
    } else {
      write_normal(r, base, count);
    }
  }

  void write_uniform(uint4 r, int64_t base, int count) const {
    if constexpr (std::is_same_v<scalar_t, double>) {
      uint64_t packed[2] = {
          (static_cast<uint64_t>(r.x) << 32) | r.y,
          (static_cast<uint64_t>(r.z) << 32) | r.w};
      for (int j = 0; j < count; j++) {
        output_[base + j] = static_cast<scalar_t>(
            transformation::uniform_real(packed[j], param0_, param1_));
      }
    } else {
      uint32_t vals[4] = {r.x, r.y, r.z, r.w};
      for (int j = 0; j < count; j++) {
        output_[base + j] = static_cast<scalar_t>(
            transformation::uniform_real(vals[j], param0_, param1_));
      }
    }
  }

  void write_normal(uint4 r, int64_t base, int count) const {
    if constexpr (std::is_same_v<scalar_t, double>) {
      auto normals = box_muller_double(r);
      double vals[2] = {normals.x, normals.y};
      for (int j = 0; j < count; j++) {
        output_[base + j] = static_cast<scalar_t>(vals[j] * param1_ + param0_);
      }
    } else {
      auto normals = box_muller_float(r);
      float vals[4] = {normals.x, normals.y, normals.z, normals.w};
      for (int j = 0; j < count; j++) {
        output_[base + j] = static_cast<scalar_t>(vals[j] * param1_ + param0_);
      }
    }
  }

  scalar_t* output_;
  const uint64_t* key_;
  int64_t num_elems_;
  param_t param0_; // low or mean
  param_t param1_; // high or stddev
};

// --- Distribution dispatch ---

void philox_distribution_validate(
    const char* op_name,
    const Tensor& self,
    const Tensor& key) {
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

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      self.scalar_type(),
      is_uniform ? "_philox_uniform_" : "_philox_normal_",
      [&] {
        using param_t =
            typename PhiloxSingleKeyFunctor<scalar_t, is_uniform>::param_t;
        constexpr int epc = elems_per_call<scalar_t>;
        int64_t num_chunks = ceil_div(self.numel(), static_cast<int64_t>(epc));
        constexpr int64_t work_group_size =
            256; // TODO: Investigate impact of wg_size 256 on XPU performance.
        // The kernel is grid-strided, so cap the launched work items at what
        // the device can keep resident instead of one work item per chunk.
        const int64_t work_items =
            std::min(num_chunks, syclMaxWorkItemsPerTile());
        const int64_t work_group_num = ceil_div(work_items, work_group_size);

        auto functor = PhiloxSingleKeyFunctor<scalar_t, is_uniform>(
            output.mutable_data_ptr<scalar_t>(),
            key_contig.const_data_ptr<uint64_t>(),
            self.numel(),
            static_cast<param_t>(param0),
            static_cast<param_t>(param1));

        sycl_kernel_submit(
            sycl::range<1>(work_group_num * work_group_size),
            sycl::range<1>(work_group_size),
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
  philox_distribution_validate("_philox_uniform_", self, key);
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
  philox_distribution_validate("_philox_normal_", self, key);
  if (self.numel() > 0) {
    philox_distribution_launch</*is_uniform=*/false>(self, key, mean, stddev);
  }
  return self;
}

} // namespace at::native::xpu
