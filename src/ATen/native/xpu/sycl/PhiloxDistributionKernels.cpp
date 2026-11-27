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

#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ceil_div.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/PhiloxStatelessRNG.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <ATen/native/xpu/sycl/Philox4x32.h>
#include <ATen/native/xpu/sycl/PhiloxDistributionKernels.h>
#include <c10/util/SmallVector.h>
#include <comm/SYCLContext.h>

#include <limits>
#include <optional>
#include <type_traits>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_philox_normal_native.h>
#include <ATen/ops/_philox_randint_native.h>
#include <ATen/ops/_philox_uniform_native.h>
#endif

namespace at::native::xpu {

// --- Box-Muller normal transforms ---

template <typename scalar_t>
inline sycl::vec<scalar_t, at::native::elems_per_call<scalar_t>> box_muller(
    uint4 r) {
  static_assert(
      std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>);
  constexpr int size = at::native::elems_per_call<scalar_t>;
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

inline uint4 philox_4x32(uint64_t seed, uint64_t offset) {
  const uint2 philox_key = {
      static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
  // Subsequence half is fixed to 0 so the 64-bit offset alone addresses
  // the whole counter space, keeping values consistent across devices.
  const uint4 counter = {
      static_cast<uint32_t>(offset), static_cast<uint32_t>(offset >> 32), 0, 0};
  return philox4x32_10(counter, philox_key);
}

// 8-byte types pack the four uint32 Philox outputs into two 64-bit values; all
// narrower types take one raw uint32 each.
template <typename scalar_t>
struct PhiloxRawSample {
  auto operator()(uint64_t seed, uint64_t offset) const {
    const uint4 r = philox_4x32(seed, offset);
    if constexpr (sizeof(scalar_t) == 8) {
      return sycl::vec<uint64_t, 2>(
          (static_cast<uint64_t>(r.x) << 32) | r.y,
          (static_cast<uint64_t>(r.z) << 32) | r.w);
    } else {
      return sycl::vec<uint32_t, 4>(r.x, r.y, r.z, r.w);
    }
  }
};

template <typename compute_t>
struct PhiloxNormalSample {
  auto operator()(uint64_t seed, uint64_t offset) const {
    return box_muller<compute_t>(philox_4x32(seed, offset));
  }
};

template <typename scalar_t>
struct PhiloxUniformParam {
  template <typename rand_t>
  scalar_t operator()(rand_t rand, int64_t /*idx*/) const {
    if constexpr (
        std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, double>) {
      const auto unit = transformation::uniform_real(
          rand, static_cast<scalar_t>(0), static_cast<scalar_t>(1));
      // Preserve CPU's intermediate rounding instead of contracting the FMA.
      volatile scalar_t scaled = unit * (hi_ - lo_);
      return scaled + lo_;
    } else {
      return static_cast<scalar_t>(
          transformation::uniform_real(rand, lo_, hi_));
    }
  }

  PhiloxUniformParam(scalar_t lo, scalar_t hi) : lo_(lo), hi_(hi) {}

 private:
  scalar_t lo_;
  scalar_t hi_;
};

template <typename scalar_t, typename compute_t>
struct PhiloxNormalParam {
  scalar_t operator()(compute_t rand, int64_t /*idx*/) const {
    return static_cast<scalar_t>(rand * stddev_ + mean_);
  }

  PhiloxNormalParam(compute_t mean, compute_t stddev)
      : mean_(mean), stddev_(stddev) {}

 private:
  compute_t mean_;
  compute_t stddev_;
};

template <typename scalar_t>
struct PhiloxRandintFullParam {
  template <typename rand_t>
  scalar_t operator()(rand_t rand, int64_t /*idx*/) const {
    return static_cast<scalar_t>(rand);
  }
};

template <typename scalar_t, typename u_t>
struct PhiloxRandintBoundedParam {
  template <typename rand_t>
  scalar_t operator()(rand_t rand, int64_t /*idx*/) const {
    return static_cast<scalar_t>(
        static_cast<u_t>(lo_ + static_cast<u_t>(rand) % range_));
  }

  PhiloxRandintBoundedParam(u_t lo, u_t range) : lo_(lo), range_(range) {}

 private:
  u_t lo_;
  u_t range_;
};

// One kernel function covers both of CUDA's key kernels:
// TrivialOffsetCalculator selects the single-key path and OffsetCalculator the
// batched-key path.
template <
    typename scalar_t,
    typename sample_t,
    typename param_t,
    typename key_offset_calc_t>
SYCL_EXT_ONEAPI_FUNCTION_PROPERTY((syclexp::nd_range_kernel<1>))
void philox_distribution_kernel_impl(
    scalar_t* output,
    const uint64_t* keys,
    int64_t elems_per_key,
    int64_t chunks_per_key,
    int64_t total_chunks,
    key_offset_calc_t key_offset_calc,
    sample_t sample_func,
    param_t param_func) {
  auto item = syclext::this_work_item::get_nd_item<1>();
  constexpr int epc = at::native::elems_per_call<scalar_t>;
  XPU_KERNEL_LOOP_TYPE(item, index, total_chunks, int64_t) {
    const int64_t key_idx = index / chunks_per_key;
    const int64_t chunk = index % chunks_per_key;
    const auto key_offset = key_offset_calc.get(key_idx)[0];
    const auto key_vec = memory::ld_vec<16>(keys + key_offset);
    const auto* key_vals = reinterpret_cast<const uint64_t*>(&key_vec);
    const uint64_t seed = key_vals[0];
    const uint64_t offset = key_vals[1];
    const int64_t chunk_offset = chunk * epc;
    const int64_t base = key_idx * elems_per_key + chunk_offset;
    const int64_t remaining = elems_per_key - chunk_offset;
    const int count = static_cast<int>(remaining < epc ? remaining : epc);

    const uint64_t chunk_seq = offset + static_cast<uint64_t>(chunk);
    const auto sample = sample_func(seed, chunk_seq);
    for (int j = 0; j < count; j++) {
      output[base + j] = param_func(sample[j], base + j);
    }
  }
}

void philox_distribution_validate(
    const char* op_name,
    const Tensor& self,
    const Tensor& key) {
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

template <
    typename scalar_t,
    typename sample_t,
    typename param_t,
    typename key_offset_calc_t>
void submit_philox_kernel(
    Tensor& output,
    const Tensor& key,
    int64_t num_keys,
    int64_t elems_per_key,
    key_offset_calc_t key_offset_calc,
    sample_t sample_func,
    param_t param_func) {
  constexpr int64_t epc = at::native::elems_per_call<scalar_t>;
  const int64_t chunks_per_key = at::ceil_div(elems_per_key, epc);
  const int64_t total_chunks = num_keys * chunks_per_key;
  constexpr int64_t work_group_size =
      256; // TODO: Investigate impact of wg_size 256 on XPU performance.
  const int64_t work_group_num =
      xpuKernelLoopGroupRange(total_chunks, work_group_size);

  constexpr auto kptr = philox_distribution_kernel_impl<
      scalar_t,
      sample_t,
      param_t,
      key_offset_calc_t>;
  sycl_kernel_submit<kptr>(
      work_group_num * work_group_size,
      work_group_size,
      at::xpu::getCurrentSYCLQueue(),
      0,
      output.mutable_data_ptr<scalar_t>(),
      key.const_data_ptr<uint64_t>(),
      elems_per_key,
      chunks_per_key,
      total_chunks,
      key_offset_calc,
      sample_func,
      param_func);
}

template <typename scalar_t, typename sample_t, typename param_t>
void philox_distribution_kernel(
    const char* op_name,
    Tensor& self,
    const Tensor& key,
    sample_t sample_func,
    param_t param_func) {
  philox_distribution_validate(op_name, self, key);
  if (self.numel() == 0) {
    return;
  }

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

  if (key.dim() == 1) {
    submit_philox_kernel<scalar_t>(
        output,
        key_contig,
        num_keys,
        elems_per_key,
        TrivialOffsetCalculator<1, int64_t>(),
        sample_func,
        param_func);
  } else {
    c10::SmallVector<int64_t, MAX_DIMS> key_sizes(key_dims);
    c10::SmallVector<int64_t, MAX_DIMS> key_strides(key_dims);
    for (int64_t i = 0; i < key_dims; i++) {
      const int64_t dim = key_dims - 1 - i;
      key_sizes[i] = self.size(dim);
      key_strides[i] = key_contig.size(dim) > 1 ? key_contig.stride(dim) : 0;
    }
    const int64_t* key_strides_ptr = key_strides.data();
    const auto key_offset_calc = OffsetCalculator<1, int64_t>(
        key_dims, key_sizes.data(), &key_strides_ptr);
    submit_philox_kernel<scalar_t>(
        output,
        key_contig,
        num_keys,
        elems_per_key,
        key_offset_calc,
        sample_func,
        param_func);
  }

  if (output.data_ptr() != self.data_ptr()) {
    self.copy_(output);
  }
}

Tensor& _philox_uniform_xpu_(
    Tensor& self,
    const Tensor& key,
    double low,
    double high) {
  TORCH_CHECK(
      self.is_floating_point(),
      "_philox_uniform_: self must be a floating point tensor, got ",
      self.scalar_type());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_uniform_", [&] {
        philox_distribution_kernel<scalar_t>(
            "_philox_uniform_",
            self,
            key,
            PhiloxRawSample<scalar_t>(),
            PhiloxUniformParam<scalar_t>(
                static_cast<scalar_t>(low), static_cast<scalar_t>(high)));
      });
  return self;
}

Tensor& _philox_normal_xpu_(
    Tensor& self,
    const Tensor& key,
    double mean,
    double stddev) {
  TORCH_CHECK(
      self.is_floating_point(),
      "_philox_normal_: self must be a floating point tensor, got ",
      self.scalar_type());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, self.scalar_type(), "_philox_normal_", [&] {
        using compute_t =
            std::conditional_t<std::is_same_v<scalar_t, double>, double, float>;
        philox_distribution_kernel<scalar_t>(
            "_philox_normal_",
            self,
            key,
            PhiloxNormalSample<compute_t>(),
            PhiloxNormalParam<scalar_t, compute_t>(
                static_cast<compute_t>(mean), static_cast<compute_t>(stddev)));
      });
  return self;
}

Tensor& _philox_randint_xpu_(
    Tensor& self,
    const Tensor& key,
    std::optional<int64_t> low,
    std::optional<int64_t> high) {
  const auto st = self.scalar_type();
  TORCH_CHECK(
      isIntegralType(st, /*includeBool=*/false),
      "_philox_randint_: self must have an integer dtype, got ",
      st);

  AT_DISPATCH_V2(
      st,
      "_philox_randint_",
      AT_WRAP([&] {
        // An absent high means the dtype's exclusive upper limit; callers pass
        // it as None since max + 1 is not representable as int64 for every
        // dtype.
        constexpr int64_t dtype_low =
            static_cast<int64_t>(std::numeric_limits<scalar_t>::min());
        const bool spans_low = !low.has_value() || *low == dtype_low;
        const bool spans_high = !high.has_value();

        if (spans_low && spans_high) {
          // Full range: every value of the dtype is equally likely, so the raw
          // bits are already the answer.
          philox_distribution_kernel<scalar_t>(
              "_philox_randint_",
              self,
              key,
              PhiloxRawSample<scalar_t>(),
              PhiloxRandintFullParam<scalar_t>());
          return;
        }

        // Reduce in the sampled width, not the output width, so dtypes narrower
        // than the sample keep all of their random bits.
        using u_t =
            std::conditional_t<sizeof(scalar_t) == 8, uint64_t, uint32_t>;
        // Number of values the dtype holds, which wraps to 0 when the dtype
        // fills the sampled width. Shifting in two steps keeps this
        // well-defined.
        constexpr u_t dtype_count = (u_t{1} << (8 * sizeof(scalar_t) - 1)) << 1;
        const u_t lo = low.has_value() ? static_cast<u_t>(*low)
                                       : static_cast<u_t>(dtype_low);
        const u_t hi = high.has_value()
            ? static_cast<u_t>(*high)
            : static_cast<u_t>(static_cast<u_t>(dtype_low) + dtype_count);
        const u_t range = hi - lo;
        TORCH_CHECK(
            range != 0,
            "_philox_randint_: [low, high) must be non-empty; pass high=None "
            "for the dtype's full range");
        // Each element of a 32-bit sample draws 32 random bits, so reducing
        // them modulo a large range noticeably oversamples the low end. A
        // range that divides the sampled width evenly is exact at any size,
        // so only guard the ranges that actually skew. (0 - range) is
        // 2^32 - range, whose remainder is 2^32 % range without needing a
        // literal that does not fit.
        if constexpr (sizeof(u_t) == 4) {
          const u_t remainder = static_cast<u_t>(u_t{0} - range) % range;
          TORCH_CHECK(
              remainder == 0 ||
                  static_cast<uint64_t>(range) < at::native::kMaxRange32,
              "_philox_randint_: range (high - low = ",
              range,
              ") does not divide 2^32 evenly and is too large for the output "
              "dtype ",
              st,
              ". Each element draws only 32 random bits, so a non-dividing "
              "range >= 2^28 gives a significantly biased distribution. Use "
              "a range that divides 2^32 (a power of two), or a 64-bit dtype "
              "(torch.int64 or torch.uint64).");
        }

        philox_distribution_kernel<scalar_t>(
            "_philox_randint_",
            self,
            key,
            PhiloxRawSample<scalar_t>(),
            PhiloxRandintBoundedParam<scalar_t, u_t>(lo, range));
      }),
      AT_EXPAND(AT_INTEGRAL_TYPES_V2));
  return self;
}

} // namespace at::native::xpu
