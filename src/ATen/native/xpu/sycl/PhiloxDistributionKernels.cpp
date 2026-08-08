/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/native/xpu/sycl/PhiloxDistributionKernels.h>
#include <c10/util/SmallVector.h>
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

template <typename scalar_t>
inline sycl::vec<scalar_t, elems_per_call<scalar_t>> box_muller(uint4 r) {
  static_assert(
      std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>);
  constexpr int size = elems_per_call<scalar_t>;
  constexpr scalar_t M = static_cast<scalar_t>(2.3283064365386963e-10);
  constexpr scalar_t M_SQUARED = M * M;
  constexpr scalar_t TWO_PI = static_cast<scalar_t>(6.2831853071795864);
  sycl::vec<scalar_t, size> uniform;
  if constexpr (std::is_same_v<scalar_t, double>) {
    for (int i = 0; i < size; i++) {
      uniform[i] = sycl::fma(
          static_cast<scalar_t>(r.val[2 * i]),
          M,
          static_cast<scalar_t>(r.val[2 * i + 1]) * M_SQUARED +
              M_SQUARED * static_cast<scalar_t>(0.5));
    }
  } else {
    for (int i = 0; i < size; i++) {
      uniform[i] = sycl::fma(
          static_cast<scalar_t>(r.val[i]), M, M * static_cast<scalar_t>(0.5));
    }
  }

  sycl::vec<scalar_t, size> result;
  for (int i = 0; i < size; i += 2) {
    const scalar_t radius =
        sycl::sqrt(static_cast<scalar_t>(-2.0) * sycl::log(uniform[i]));
    const scalar_t angle = TWO_PI * uniform[i + 1];
    result[i] = radius * sycl::cos(angle);
    result[i + 1] = radius * sycl::sin(angle);
  }
  return result;
}

template <typename scalar_t, bool is_uniform, typename key_offset_calc_t>
struct PhiloxDistributionFunctor {
  // Uniform masks the raw bits against the output dtype's mantissa, so its
  // bounds stay in scalar_t. Normal is transformed in compute precision.
  using param_t = std::conditional_t<
      is_uniform,
      scalar_t,
      std::conditional_t<std::is_same_v<scalar_t, double>, double, float>>;

  void operator()(sycl::nd_item<1> item) const {
    constexpr int epc = elems_per_call<scalar_t>;
    XPU_KERNEL_LOOP_TYPE(item, index, total_chunks_, int64_t) {
      const int64_t key_idx = index / chunks_per_key_;
      const int64_t chunk = index % chunks_per_key_;
      const auto key_offset = key_offset_calc_.get(key_idx)[0];
      const auto key_vec = memory::ld_vec<16>(keys_ + key_offset);
      const auto* key_vals = reinterpret_cast<const uint64_t*>(&key_vec);
      const uint64_t seed = key_vals[0];
      const uint64_t offset = key_vals[1];
      const int64_t chunk_offset = chunk * epc;
      const int64_t base = key_idx * elems_per_key_ + chunk_offset;
      const int64_t remaining = elems_per_key_ - chunk_offset;
      const int count = static_cast<int>(remaining < epc ? remaining : epc);
      const auto r = philox_4x32(seed, offset + static_cast<uint64_t>(chunk));
      write_values(r, base, count);
    }
  }

  PhiloxDistributionFunctor(
      scalar_t* output,
      const uint64_t* keys,
      int64_t elems_per_key,
      int64_t chunks_per_key,
      int64_t total_chunks,
      key_offset_calc_t key_offset_calc,
      param_t param0,
      param_t param1)
      : output_(output),
        keys_(keys),
        elems_per_key_(elems_per_key),
        chunks_per_key_(chunks_per_key),
        total_chunks_(total_chunks),
        key_offset_calc_(key_offset_calc),
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
      const uint64_t packed[2] = {
          (static_cast<uint64_t>(r.x) << 32) | r.y,
          (static_cast<uint64_t>(r.z) << 32) | r.w};
      for (int j = 0; j < count; j++) {
        output_[base + j] = transform_uniform(packed[j]);
      }
    } else {
      const uint32_t vals[4] = {r.x, r.y, r.z, r.w};
      for (int j = 0; j < count; j++) {
        output_[base + j] = transform_uniform(vals[j]);
      }
    }
  }

  template <typename rand_t>
  scalar_t transform_uniform(rand_t rand) const {
    if constexpr (
        std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>) {
      const auto unit = transformation::uniform_real(
          rand, static_cast<scalar_t>(0), static_cast<scalar_t>(1));
      // Preserve CPU's intermediate rounding instead of contracting the FMA.
      volatile scalar_t scaled = unit * (param1_ - param0_);
      return scaled + param0_;
    } else {
      return static_cast<scalar_t>(
          transformation::uniform_real(rand, param0_, param1_));
    }
  }

  void write_normal(uint4 r, int64_t base, int count) const {
    using compute_t =
        std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
    const auto normals = box_muller<compute_t>(r);
    for (int j = 0; j < count; j++) {
      output_[base + j] = static_cast<scalar_t>(normals[j] * param1_ + param0_);
    }
  }

  scalar_t* output_;
  const uint64_t* keys_;
  int64_t elems_per_key_;
  int64_t chunks_per_key_;
  int64_t total_chunks_;
  key_offset_calc_t key_offset_calc_;
  param_t param0_; // low or mean
  param_t param1_; // high or stddev
};

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
}

template <typename scalar_t, bool is_uniform, typename key_offset_calc_t>
void philox_distribution_kernel(
    Tensor& output,
    const Tensor& key,
    int64_t num_keys,
    int64_t elems_per_key,
    key_offset_calc_t key_offset_calc,
    double param0,
    double param1) {
  using functor_t =
      PhiloxDistributionFunctor<scalar_t, is_uniform, key_offset_calc_t>;
  using param_t = typename functor_t::param_t;
  constexpr int64_t epc = elems_per_call<scalar_t>;
  const int64_t chunks_per_key = at::ceil_div(elems_per_key, epc);
  const int64_t total_chunks = num_keys * chunks_per_key;
  constexpr int64_t work_group_size =
      256; // TODO: Investigate impact of wg_size 256 on XPU performance.
  const int64_t work_group_num =
      xpuKernelLoopGroupRange(total_chunks, work_group_size);

  const auto functor = functor_t(
      output.mutable_data_ptr<scalar_t>(),
      key.const_data_ptr<uint64_t>(),
      elems_per_key,
      chunks_per_key,
      total_chunks,
      key_offset_calc,
      static_cast<param_t>(param0),
      static_cast<param_t>(param1));

  sycl_kernel_submit(
      sycl::range<1>(work_group_num * work_group_size),
      sycl::range<1>(work_group_size),
      at::xpu::getCurrentSYCLQueue(),
      functor);
}

template <bool is_uniform>
void philox_distribution_launch(
    Tensor& self,
    const Tensor& key,
    double param0,
    double param1) {
  auto output = self.contiguous();
  const auto key_contig = key.contiguous();

  int64_t elems_per_key = 1;
  int64_t key_dims = self.dim();
  if (key.dim() == 1) {
    elems_per_key = self.numel();
    key_dims = 0;
  } else {
    for (int64_t dim = self.dim() - 1; dim >= 0; dim--) {
      if (key.size(dim) != 1) {
        break;
      }
      elems_per_key *= self.size(dim);
      key_dims--;
    }
  }
  const int64_t num_keys = self.numel() / elems_per_key;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      self.scalar_type(),
      is_uniform ? "_philox_uniform_" : "_philox_normal_",
      [&] {
        if (key.dim() == 1) {
          philox_distribution_kernel<scalar_t, is_uniform>(
              output,
              key_contig,
              num_keys,
              elems_per_key,
              TrivialOffsetCalculator<1, int64_t>(),
              param0,
              param1);
        } else {
          c10::SmallVector<int64_t, MAX_DIMS> key_sizes(key_dims);
          c10::SmallVector<int64_t, MAX_DIMS> key_strides(key_dims);
          for (int64_t i = 0; i < key_dims; i++) {
            const int64_t dim = key_dims - 1 - i;
            key_sizes[i] = self.size(dim);
            key_strides[i] =
                key_contig.size(dim) > 1 ? key_contig.stride(dim) : 0;
          }
          const int64_t* key_strides_ptr = key_strides.data();
          const auto key_offset_calc = OffsetCalculator<1, int64_t>(
              key_dims, key_sizes.data(), &key_strides_ptr);
          philox_distribution_kernel<scalar_t, is_uniform>(
              output,
              key_contig,
              num_keys,
              elems_per_key,
              key_offset_calc,
              param0,
              param1);
        }
      });

  if (output.data_ptr() != self.data_ptr()) {
    self.copy_(output);
  }
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
