/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/OpMathType.h>
#include <ATen/ceil_div.h>
#include <ATen/core/TensorBody.h>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorIteratorDynamicCasting.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>

#include <ATen/native/xpu/sycl/ElementwiseInvoke.h>
#include <ATen/native/xpu/sycl/MemoryAccess.h>
#include <ATen/native/xpu/sycl/OffsetCalculator.h>
#include <comm/SYCLContext.h>

using namespace at::xpu;

namespace at {
namespace native {
namespace xpu {

template <int item_work_size, typename func_t, typename policy_t>
inline void elementwise_kernel_helper(func_t f, policy_t policy) {
  using traits = function_traits<func_t>;
  using return_t = typename traits::result_type;
  using args_t = typename traits::ArgsTuple;

  return_t results[item_work_size];
  args_t args[item_work_size];

  // load
  policy.load(args);

  // compute
#pragma unroll
  for (int i = 0; i < item_work_size; i++) {
    if (policy.check_inbounds(i)) {
      results[i] = std::apply(f, args[i]);
    }
  }

  // store
  policy.store(results);
}

template <
    typename func_t,
    typename array_t,
    typename in_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
struct UnrolledElementwiseKernel {
  static constexpr int item_work_size = 4;

  void operator()(sycl::nd_item<1> item) const {
    int grpsz = item.get_local_range(0);
    int grpid = item.get_group(0);
    int lid = item.get_local_id(0);
    int remaining = numel_ - item_work_size * grpsz * grpid;
    auto policy = at::native::memory::policies::unroll<
        item_work_size,
        array_t,
        in_calc_t,
        out_calc_t,
        loader_t,
        storer_t>(data_, remaining, ic_, oc_, l_, s_, lid, grpid, grpsz);
    elementwise_kernel_helper<item_work_size>(f_, policy);
  };

  UnrolledElementwiseKernel(
      int numel,
      func_t f,
      array_t data,
      in_calc_t ic,
      out_calc_t oc,
      loader_t l,
      storer_t s)
      : numel_(numel), f_(f), data_(data), ic_(ic), oc_(oc), l_(l), s_(s) {}

 private:
  int numel_;
  func_t f_;
  array_t data_;
  in_calc_t ic_;
  out_calc_t oc_;
  loader_t l_;
  storer_t s_;
};

// Reconstruct only values whose type completely determines their state.
template <typename func_t>
constexpr bool is_stateless_functor_v =
    std::is_empty_v<func_t> && std::is_default_constructible_v<func_t>;

template <typename calc_t>
constexpr bool is_trivial_offset_calculator_v = false;
template <int N, typename index_t>
constexpr bool
    is_trivial_offset_calculator_v<TrivialOffsetCalculator<N, index_t>> = true;

namespace launch_config {
// Wide and integral inner broadcasts fix their subgroup size.
constexpr int broadcast_work_group_size = 256;
constexpr int broadcast_wide_work_group_size = 128;
constexpr int broadcast_wide_sub_group_size = 16;

constexpr int mixed_dtype_sub_group_size = 16;
} // namespace launch_config

template <
    int vec_size,
    typename func_t,
    typename array_t,
    typename in_calc_t,
    bool full_tile,
    int work_group_size = 0>
struct VectorizedElementwiseKernel {
  template <bool omit, typename T>
  static auto capture(const T& value) {
    if constexpr (omit) {
      return std::tuple<>{};
    } else {
      return value;
    }
  }

  template <typename T, typename stored_t>
  static T restore(const stored_t& value) {
    if constexpr (std::is_same_v<T, stored_t>) {
      return value;
    } else {
      return T{};
    }
  }

  template <int width>
  static inline void apply(
      const func_t& f,
      array_t data,
      in_calc_t ic,
      uint32_t lid,
      uint32_t grpid,
      uint32_t grpsz) {
    auto policy = memory::policies::vectorized<width, array_t, in_calc_t>(
        data, ic, lid, grpid, grpsz);
    elementwise_kernel_helper<width>(f, policy);
  }

  void operator()(sycl::nd_item<1> item) const {
    const uint32_t grpsz =
        work_group_size ? work_group_size : item.get_local_range(0);
    const uint32_t grpid = item.get_group(0);
    const uint32_t lid = item.get_local_id(0);
    const auto f = restore<func_t>(f_);
    const auto ic = restore<in_calc_t>(ic_);
    if constexpr (full_tile) {
      apply<vec_size>(f, data_, ic, lid, grpid, grpsz);
    } else {
      const uint32_t numel = numel_;
      if (grpid * grpsz + lid < numel / vec_size) {
        apply<vec_size>(f, data_, ic, lid, grpid, grpsz);
      }
      if (grpid == numel / (vec_size * grpsz) && lid < numel % vec_size) {
        apply<1>(f, data_, ic, (numel / vec_size) * vec_size + lid, 0, 1);
      }
    }
  }

  VectorizedElementwiseKernel(
      int numel,
      const func_t& f,
      array_t data,
      in_calc_t ic)
      : numel_(capture<full_tile>(numel)),
        f_(capture<is_stateless_functor_v<func_t>>(f)),
        data_(data),
        ic_(capture<is_trivial_offset_calculator_v<in_calc_t>>(ic)) {}

 private:
  [[no_unique_address]] std::conditional_t<full_tile, std::tuple<>, int> numel_;
  [[no_unique_address]] std::
      conditional_t<is_stateless_functor_v<func_t>, std::tuple<>, func_t>
          f_;
  array_t data_;
  [[no_unique_address]] std::conditional_t<
      is_trivial_offset_calculator_v<in_calc_t>,
      std::tuple<>,
      in_calc_t>
      ic_;
};

template <
    int num_outputs,
    typename func_t,
    typename array_t,
    typename in_calc_t,
    typename out_calc_t>
struct UnrolledElementwiseForMultiOutputsKernel {
  static constexpr int item_work_size = 4;

  void operator()(sycl::nd_item<1> item_id) const {
    int grpsz = item_id.get_local_range(0);
    int grpid = item_id.get_group(0);
    int lid = item_id.get_local_id(0);
    int remaining = numel_ - item_work_size * grpsz * grpid;
    auto policy = at::native::memory::policies::multi_outputs_unroll<
        item_work_size,
        array_t,
        in_calc_t,
        out_calc_t,
        num_outputs>(data_, remaining, ic_, oc_, lid, grpid, grpsz);
    elementwise_kernel_helper<item_work_size>(f_, policy);
  };

  UnrolledElementwiseForMultiOutputsKernel(
      int numel,
      func_t f,
      array_t data,
      in_calc_t ic,
      out_calc_t oc)
      : numel_(numel), f_(f), data_(data), ic_(ic), oc_(oc) {}

 private:
  int numel_;
  func_t f_;
  array_t data_;
  in_calc_t ic_;
  out_calc_t oc_;
};

template <
    int num_outputs,
    typename func_t,
    typename array_t,
    typename in_calc_t,
    typename out_calc_t>
struct UnrolledElementwiseKernelForMultiOutputs {
  static constexpr int item_work_size = 4;

  void operator()(sycl::nd_item<1> item) const {
    int grpsz = item.get_local_range(0);
    int grpid = item.get_group(0);
    int lid = item.get_local_id(0);
    int remaining = numel_ - item_work_size * grpsz * grpid;
    auto policy = at::native::memory::policies::multi_outputs_unroll<
        item_work_size,
        array_t,
        in_calc_t,
        out_calc_t,
        num_outputs>(data_, remaining, ic_, oc_, lid, grpid, grpsz);
    elementwise_kernel_helper<item_work_size>(f_, policy);
  };

  UnrolledElementwiseKernelForMultiOutputs(
      int numel,
      func_t f,
      array_t data,
      in_calc_t ic,
      out_calc_t oc)
      : numel_(numel), f_(f), data_(data), ic_(ic), oc_(oc) {}

 private:
  int numel_;
  func_t f_;
  array_t data_;
  in_calc_t ic_;
  out_calc_t oc_;
};

template <int vec_size, typename func_t>
struct ElementwiseGroupRangeKernel {
  void operator()(sycl::nd_item<1> item) const {
    int wg_sz = item.get_local_range(0);
    int group_work_size = wg_sz * vec_size;
    int idx = group_work_size * item.get_group(0) + item.get_local_id(0);
#pragma unroll
    for (int i = 0; i < vec_size; i++) {
      if (idx < numel_) {
        f_(idx);
        idx += wg_sz;
      }
    }
  };

  ElementwiseGroupRangeKernel(int numel, func_t f) : numel_(numel), f_(f) {}

 private:
  int numel_;
  func_t f_;
};

template <typename func_t>
struct ElementwiseGlobalRangeKernel {
  void operator()(sycl::nd_item<1> item) const {
    int linear_idx =
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);
    for (int idx = linear_idx; idx < numel_;
         idx += item.get_group_range(0) * item.get_local_range(0)) {
      if (idx < numel_) {
        f_(idx);
      }
    }
  };

  ElementwiseGlobalRangeKernel(int numel, func_t f) : numel_(numel), f_(f) {}

 private:
  int numel_;
  func_t f_;
};

template <
    typename arg0_t,
    int ntensors,
    typename offset_calc_t,
    typename func_t>
struct LegacyKernelScalarFunctor {
  void operator()(int idx) const {
    auto offsets = offset_calc_.get(idx);
    arg0_t* out = (arg0_t*)(data_[0] + offsets[0]);
    *out = invoke(f_, &data_.data[1], &offsets.data[1], 1);
  }

  LegacyKernelScalarFunctor(
      at::detail::Array<char*, ntensors> data,
      offset_calc_t offset_calc,
      const func_t f)
      : data_(data), offset_calc_(offset_calc), f_(f) {}

 private:
  at::detail::Array<char*, ntensors> data_;
  offset_calc_t offset_calc_;
  const func_t f_;
};

template <
    typename arg0_t,
    int ntensors,
    typename offset_calc_t,
    typename func_t>
struct LegacyKernelWithCastScalarFunctor {
  void operator()(int idx) const {
    auto offsets = offset_calc_.get(idx);
    void* out = data_[0] + offsets[0];
    arg0_t result =
        invoke(f_, &data_.data[1], &offsets.data[1], &dtypes_.data[1], 1);
    c10::cast_and_store<arg0_t>(dtypes_[0], out, result);
  }

  LegacyKernelWithCastScalarFunctor(
      at::detail::Array<char*, ntensors> data,
      at::detail::Array<ScalarType, ntensors> dtypes,
      offset_calc_t offset_calc,
      const func_t f)
      : data_(data), dtypes_(dtypes), offset_calc_(offset_calc), f_(f) {}

 private:
  at::detail::Array<char*, ntensors> data_;
  at::detail::Array<ScalarType, ntensors> dtypes_;
  offset_calc_t offset_calc_;
  const func_t f_;
};

template <int vec_size, typename func_t>
static void launch_legacy_group_range_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  auto ker = ElementwiseGroupRangeKernel<vec_size, func_t>(N, f);

  int64_t wg_sz = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();
  int64_t num_wg = ceil_div<int64_t>(N, wg_sz * vec_size);
  sycl_kernel_submit(wg_sz * num_wg, wg_sz, getCurrentSYCLQueue(), ker);
}

template <typename func_t>
static void launch_legacy_global_range_kernel(int64_t N, const func_t& f) {
  TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
  if (N == 0) {
    return;
  }

  auto ker = ElementwiseGlobalRangeKernel<func_t>(N, f);

  int64_t wg_sz = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();
  int64_t num_wg = ceil_div<int64_t>(N, wg_sz);
  int64_t hw_max_num_wg = at::xpu::getDeviceMaxWorkItems() / wg_sz;
  num_wg = num_wg > hw_max_num_wg ? hw_max_num_wg : num_wg;
  sycl_kernel_submit(wg_sz * num_wg, wg_sz, getCurrentSYCLQueue(), ker);
}

template <
    typename func_t,
    typename array_t,
    typename in_calc_t,
    typename out_calc_t,
    typename loader_t,
    typename storer_t>
static inline void launch_unrolled_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    in_calc_t ic,
    out_calc_t oc,
    loader_t l,
    storer_t s) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  auto ker = UnrolledElementwiseKernel(N, f, data, ic, oc, l, s);
  using ker_t = decltype(ker);

  int64_t wg_sz = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();
  int64_t num_wg = ceil_div<int64_t>(N, wg_sz * ker_t::item_work_size);
  sycl_kernel_submit(wg_sz * num_wg, wg_sz, getCurrentSYCLQueue(), ker);
}

constexpr int max_scalar_size_(std::tuple<>) {
  return 0;
}

template <typename scalar_t, typename... types>
constexpr int max_scalar_size_(std::tuple<scalar_t, types...>) {
  return std::max<int>(
      sizeof(scalar_t), max_scalar_size_(std::tuple<types...>{}));
}

template <typename func_t>
constexpr static inline int max_scalar_size() {
  using traits = function_traits<func_t>;
  using args_t = typename traits::ArgsTuple;
  constexpr auto size = max_scalar_size_(args_t{});
  using return_t = typename traits::result_type;
  return std::max<int>(sizeof(return_t), size);
}

template <int vec_size, bool unary, typename func_t, typename in_calc_t>
class StridedElementwiseKernel;

// Density is relative to TI's reordered/coalesced iteration dimensions, not
// the tensor's logical memory format. Alignment remains path-specific.
static inline bool has_dense_output(
    const TensorIteratorBase& iter,
    int64_t element_size) {
  int64_t expected = element_size;
  for (int dim = 0; dim < iter.ndim(); ++dim) {
    if (iter.shape()[dim] != 1 && iter.strides(0)[dim] != expected) {
      return false;
    }
    expected *= iter.shape()[dim];
  }
  return true;
}

// TI normally rejects detectable internal/partial overlap, but permits exact
// in-place aliasing and may report TooHard for strided views. The binary
// broadcast paths additionally reject overlapping input/output bounding ranges.
// This is a conservative fast-path guard, not a general alias-safety proof.
// The caller must establish that the output is dense in iteration order.
static inline bool overlaps_dense_output(
    const TensorIteratorBase& iter,
    const void* output,
    uintptr_t input_begin,
    uintptr_t input_end) {
  const auto output_begin = reinterpret_cast<uintptr_t>(output);
  const auto output_end = output_begin + iter.numel() * iter.element_size(0);
  return input_begin < output_end && output_begin < input_end;
}

static inline bool broadcast_overlaps_output(
    const TensorIteratorBase& iter,
    const void* output,
    int input,
    bool broadcast_only = false) {
  const auto strides = iter.strides(input);
  const auto input_begin = reinterpret_cast<uintptr_t>(iter.data_ptr(input));
  auto input_end = input_begin + iter.element_size(input);
  bool broadcast = !broadcast_only;
  for (int dim = 0; dim < iter.ndim(); ++dim) {
    broadcast |= iter.shape()[dim] > 1 && strides[dim] == 0;
    input_end += (iter.shape()[dim] - 1) * strides[dim];
  }
  return broadcast &&
      overlaps_dense_output(iter, output, input_begin, input_end);
}

template <typename func_t, typename array_t, typename in_calc_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    in_calc_t input_calc,
    int vec_size) {
  constexpr auto max_scalar_bytes = max_scalar_size<func_t>();
  constexpr int specialized_wg_size = 256;
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  int64_t wg_sz = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();

#define VEC_KER(vec_size)                                           \
  {                                                                 \
    TORCH_CHECK(max_scalar_bytes* vec_size <= 16);                  \
    if constexpr (max_scalar_bytes * vec_size <= 16) {              \
      auto submit = [&](auto group_size) {                          \
        auto launch = [&](auto full_tile) {                         \
          auto ker = VectorizedElementwiseKernel<                   \
              vec_size,                                             \
              func_t,                                               \
              array_t,                                              \
              in_calc_t,                                            \
              decltype(full_tile)::value,                           \
              decltype(group_size)::value>(N, f, data, input_calc); \
          int64_t num_wg = ceil_div<int64_t>(N, wg_sz * vec_size);  \
          sycl_kernel_submit(                                       \
              wg_sz* num_wg, wg_sz, getCurrentSYCLQueue(), ker);    \
        };                                                          \
        if (N % (wg_sz * vec_size) == 0) {                          \
          launch(std::true_type{});                                 \
        } else {                                                    \
          launch(std::false_type{});                                \
        }                                                           \
      };                                                            \
      if (wg_sz == specialized_wg_size) {                           \
        submit(std::integral_constant<int, specialized_wg_size>{}); \
      } else {                                                      \
        submit(std::integral_constant<int, 0>{});                   \
      }                                                             \
    }                                                               \
  }

  switch (vec_size) {
    case 16:
      VEC_KER(16);
      break;
    case 8:
      VEC_KER(8);
      break;
    case 4:
      VEC_KER(4);
      break;
    case 2:
      VEC_KER(2);
      break;
    case 1: {
      auto input_calc = TrivialOffsetCalculator<traits::arity>();
      auto output_calc = TrivialOffsetCalculator<1>();
      auto loader = memory::LoadWithoutCast();
      auto storer = memory::StoreWithoutCast();

      auto ker = UnrolledElementwiseKernel(
          N, f, data, input_calc, output_calc, loader, storer);
      using ker_t = decltype(ker);

      int64_t num_wg = ceil_div<int64_t>(N, wg_sz * ker_t::item_work_size);
      sycl_kernel_submit(wg_sz * num_wg, wg_sz, getCurrentSYCLQueue(), ker);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unexpected vectorization size");
  }
}

template <
    int num_outputs,
    typename func_t,
    typename array_t,
    typename in_calc_t,
    typename out_calc_t>
static inline void launch_unrolled_kernel_for_multi_outputs(
    int64_t N,
    const func_t& f,
    array_t data,
    in_calc_t ic,
    out_calc_t oc) {
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());

  auto ker = UnrolledElementwiseForMultiOutputsKernel<
      num_outputs,
      func_t,
      array_t,
      in_calc_t,
      out_calc_t>(N, f, data, ic, oc);
  using ker_t = decltype(ker);

  int64_t wg_sz = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();
  int64_t num_wg = ceil_div<int64_t>(N, ker_t::item_work_size * wg_sz);
  sycl_kernel_submit(wg_sz * num_wg, wg_sz, getCurrentSYCLQueue(), ker);
}

template <typename func_t, typename data_t>
static inline bool can_vectorize_for_non_contigouous(
    TensorIteratorBase& iter,
    const data_t& data,
    int& vec_size) {
  // Only optimizing non conitguous cases here
  if (iter.is_contiguous())
    return false;
  // Fastest moving dimension should be contiguous
  if (!iter.has_contiguous_first_dim())
    return false;
  // Output should be contiguous
  if (!iter.tensor(0).is_contiguous())
    return false;
  vec_size = memory::can_vectorize_up_to<func_t>(data);
  if (vec_size <= 1)
    return false;
  for (int i = 0; i < iter.ntensors(); i++) {
    // Checking each row has enough alignments
    auto strides = iter.strides(i);
    for (int dim = 1; dim < (int)strides.size(); dim++) {
      int64_t base_of_row = strides[dim];
      int64_t vec_size_in_bytes = strides[0] * vec_size;
      while (base_of_row % vec_size_in_bytes) {
        vec_size >>= 1;
        if (vec_size <= 1)
          return false;
      }
    }
  }
  return vec_size > 1;
}

constexpr int strided_work_group_size = 128;

// Where the inputs of the vector starting at output element `first` are, for
// BroadcastElementwiseKernel: in a one-dimensional iterator the dense input
// follows the output and the broadcast input is a single element.
template <int broadcast_arg>
struct SingleElementBroadcast {
  at::detail::Array<uint32_t, 2> get(uint32_t first) const {
    at::detail::Array<uint32_t, 2> offsets;
    offsets[broadcast_arg] = 0;
    offsets[1 - broadcast_arg] = first;
    return offsets;
  }
  uint32_t numel;
};

// Reuse the outer quotient directly for a two-dimensional binary iterator.
static inline auto strided_binary_offsets(
    const OffsetCalculator<2>& ic,
    uint32_t first) {
  if (ic.dims == 2) {
    const auto divmod = ic.sizes_[0].divmod(first);
    OffsetCalculator<2>::offset_type offsets;
#pragma unroll
    for (int arg = 0; arg < 2; ++arg) {
      offsets[arg] =
          divmod.mod * ic.strides_[0][arg] + divmod.div * ic.strides_[1][arg];
    }
    return offsets;
  }
  return ic.get(first);
}

// Otherwise each innermost row of the broadcast input is a single element.
struct InnerBroadcast {
  template <int broadcast_arg>
  OffsetCalculator<2>::offset_type get(uint32_t first) const {
    if (calc.dims != 2) {
      return calc.get(first);
    }
    const auto& divider = calc.sizes_[0];
    const uint32_t row = divider.div(first);
    constexpr int dense_arg = 1 - broadcast_arg;
    OffsetCalculator<2>::offset_type offsets;
    offsets[broadcast_arg] = row * calc.strides_[1][broadcast_arg];
    // This equals row * pitch + column, including zero outer pitch. The
    // difference and addition use unsigned offsets in element units.
    offsets[dense_arg] =
        first + row * (calc.strides_[1][dense_arg] - divider.divisor);
    return offsets;
  }
  uint32_t numel;
  OffsetCalculator<2> calc;
};

// For a binary functor whose output is dense, one input is dense along the
// innermost dimension and the other broadcasts along it (stride 0), e.g.
// Tensor op single-element XPU Tensor or [M, N] op [M, 1]. Vectors never cross
// an innermost row, so each takes one vector of the dense input and one
// element of the broadcast input. The functor and its argument order are
// unchanged. The broadcast position is a template parameter so both loads are
// issued back to back, and the pointers and the functor come first so they
// stay in the inline kernel payload. Scalar FP16/BF16 vec8 uses WG128 and
// SIMD16. Integral inner broadcasts use SIMD32, and the compiler selects SIMD
// for other widths.
// Single-element broadcasts may have an ordinary tail. With full_waves, all
// work-items after the whole waves of two-element vectors take one element
// each (see BroadcastLaunchConfig).
template <
    int vec_size,
    int broadcast_arg,
    bool small_work_group,
    typename func_t,
    typename layout_t,
    bool full_waves = false>
struct BroadcastElementwiseKernel {
  static_assert(!full_waves || vec_size == 2);
  static constexpr int work_group_size = small_work_group
      ? launch_config::broadcast_wide_work_group_size
      : launch_config::broadcast_work_group_size;
  using traits = function_traits<func_t>;
  using dense_t =
      std::decay_t<typename traits::template arg<1 - broadcast_arg>::type>;
  using broadcast_t =
      std::decay_t<typename traits::template arg<broadcast_arg>::type>;
  using return_t = typename traits::result_type;

  return_t apply(dense_t value, broadcast_t broadcast) const {
    if constexpr (broadcast_arg == 0) {
      return f_(broadcast, value);
    } else {
      return f_(value, broadcast);
    }
  }

  auto get_offsets(uint32_t first) const {
    if constexpr (std::is_same_v<layout_t, InnerBroadcast>) {
      return layout_.template get<broadcast_arg>(first);
    } else {
      return layout_.get(first);
    }
  }

  // A constant group size keeps the index free of payload loads. With
  // full_waves, layout_.numel holds the number of vectors instead of
  // elements, and the launch has exactly one work-item per later element.
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t index = uint32_t(item.get_group(0)) * work_group_size +
        uint32_t(item.get_local_id(0));
    const uint32_t vectors =
        full_waves ? layout_.numel : layout_.numel / vec_size;
    if (index < vectors) {
      const auto offsets = get_offsets(index * vec_size);
      const auto values =
          memory::load_vector<vec_size>(dense_ + offsets[1 - broadcast_arg], 0);
      const broadcast_t broadcast =
          c10::load(broadcast_ + offsets[broadcast_arg]);
      memory::aligned_vector<return_t, vec_size> results;
#pragma unroll
      for (int j = 0; j < vec_size; ++j) {
        results.val[j] = apply(values.val[j], broadcast);
      }
      reinterpret_cast<memory::aligned_vector<return_t, vec_size>*>(
          output_)[index] = results;
    } else if constexpr (
        full_waves || !std::is_same_v<layout_t, InnerBroadcast>) {
      if (full_waves || index < vectors + layout_.numel % vec_size) {
        const uint32_t k = vectors * (vec_size - 1) + index;
        const auto offsets = get_offsets(k);
        output_[k] = apply(
            c10::load(dense_ + offsets[1 - broadcast_arg]),
            c10::load(broadcast_ + offsets[broadcast_arg]));
      }
    }
  }

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    if constexpr (small_work_group) {
      return sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::sub_group_size<
              launch_config::broadcast_wide_sub_group_size>};
    } else if constexpr (
        std::is_same_v<layout_t, InnerBroadcast> &&
        std::is_integral_v<dense_t> && std::is_integral_v<broadcast_t> &&
        std::is_integral_v<return_t>) {
      return sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::sub_group_size<32>};
    } else {
      return sycl::ext::oneapi::experimental::properties{};
    }
  }

  BroadcastElementwiseKernel(
      const dense_t* dense,
      const broadcast_t* broadcast,
      return_t* output,
      const func_t& f,
      const layout_t& layout)
      : dense_(dense),
        broadcast_(broadcast),
        output_(output),
        f_(f),
        layout_(layout) {}

 private:
  const dense_t* dense_;
  const broadcast_t* broadcast_;
  return_t* output_;
  func_t f_;
  layout_t layout_;
};

// Launch configuration of BroadcastElementwiseKernel. Only the output and the
// dense input are streamed; the broadcast input takes one element per vector,
// so its type does not size the vectors (a bool mask must not widen int64
// work-items).
template <typename func_t, int broadcast_arg, bool single_element>
struct BroadcastLaunchConfig {
  using traits = function_traits<func_t>;
  using dense_t =
      std::decay_t<typename traits::template arg<1 - broadcast_arg>::type>;
  using broadcast_t =
      std::decay_t<typename traits::template arg<broadcast_arg>::type>;
  using output_t = typename traits::result_type;
  static constexpr bool same_storage = std::is_same_v<dense_t, output_t> &&
      std::is_same_v<broadcast_t, output_t>;
  static constexpr bool low_precision = same_storage &&
      (std::is_same_v<output_t, at::Half> ||
       std::is_same_v<output_t, at::BFloat16>);
  static constexpr bool float_storage =
      same_storage && std::is_same_v<output_t, float>;
  template <int vec_size>
  static constexpr bool small_work_group =
      single_element && low_precision && vec_size == 8;
  static constexpr int dense_bytes = sizeof(dense_t);
  static constexpr int output_bytes = sizeof(output_t);
  static constexpr int min_bytes = std::min(dense_bytes, output_bytes);
  static constexpr int max_bytes = std::max(dense_bytes, output_bytes);
  // Wide scalar broadcasts take 16 bytes of the widest streamed operand.
  // In the adaptive path, a work-item takes 8 bytes when there is enough work
  // for every hardware thread, twice over when 4 bytes hold several elements.
  // The narrowest operand, e.g. a bool output, keeps at least 4 bytes.
  static constexpr int narrow_vec = std::max(1, 4 / min_bytes);
  static constexpr int wide_vec = std::max(8 / max_bytes, narrow_vec);
  static constexpr int waves = narrow_vec > 1 ? 2 : 1;
  // When adaptive work changes from single elements to two-element vectors,
  // process the remaining elements individually after the full vector waves.
  static constexpr bool full_waves = narrow_vec == 1 && wide_vec == 2;
  // Broadcast rows gain from vectors only when a streamed operand is narrower
  // than 8 bytes; otherwise the generic kernel already accesses 8 bytes per
  // work-item, and its grid-stride loop suits long write streams.
  static constexpr bool vectorize_rows = min_bytes < 8;

  static bool wide(int64_t numel) {
    return numel / wide_vec >=
        waves * int64_t(at::xpu::getDeviceMaxWorkItems());
  }
};

// Pack a device scalar or a single innermost stride-zero operand.
template <typename func_t, typename array_t>
static inline bool try_launch_broadcast_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    const array_t& data,
    bool dense_output) {
  static_assert(function_traits<func_t>::arity == 2);
  const int ndim = iter.ndim();
  const int numel = iter.numel();
  if (ndim == 0 || numel < 2 || !dense_output) {
    return false;
  }
  auto launch_arg = [&](auto arg, auto single) {
    constexpr int broadcast_arg = decltype(arg)::value;
    constexpr bool single_element = decltype(single)::value;
    constexpr int dense = 2 - broadcast_arg;
    // A broadcast input stored in the output would be read while written.
    if (iter.strides(broadcast_arg + 1)[0] != 0 ||
        iter.strides(dense)[0] != iter.element_size(dense) ||
        broadcast_overlaps_output(iter, data[0], broadcast_arg + 1)) {
      return false;
    }
    using config = BroadcastLaunchConfig<func_t, broadcast_arg, single_element>;
    if constexpr (!single_element) {
      if (!config::vectorize_rows ||
          broadcast_overlaps_output(iter, data[0], dense, true)) {
        return false;
      }
    }

    auto launch_layout = [&](auto vec, const auto& layout) {
      auto submit = [&](auto whole_waves) {
        constexpr int vec_v = decltype(vec)::value;
        constexpr bool full_waves = decltype(whole_waves)::value;
        using kernel_t = BroadcastElementwiseKernel<
            vec_v,
            broadcast_arg,
            config::template small_work_group<vec_v>,
            func_t,
            std::decay_t<decltype(layout)>,
            full_waves>;
        constexpr int64_t wg_size = kernel_t::work_group_size;
        int64_t items = numel / vec_v;
        if constexpr (single_element) {
          items += numel % vec_v;
        }
        auto kernel_layout = layout;
        if constexpr (full_waves) {
          // A wide launch has at least one whole wave of vectors. A
          // multiple of the group size of work-items with numel - items vectors
          // leaves exactly one element per later work-item.
          const int64_t capacity = at::xpu::getDeviceMaxWorkItems();
          const int64_t vectors = numel / 2 / capacity * capacity;
          items = ceil_div<int64_t>(numel - vectors, wg_size) * wg_size;
          kernel_layout.numel = uint32_t(numel - items);
        }
        sycl_kernel_submit(
            ceil_div(items, wg_size) * wg_size,
            wg_size,
            getCurrentSYCLQueue(),
            kernel_t(
                reinterpret_cast<const typename kernel_t::dense_t*>(
                    data[dense]),
                reinterpret_cast<const typename kernel_t::broadcast_t*>(
                    data[broadcast_arg + 1]),
                reinterpret_cast<typename kernel_t::return_t*>(data[0]),
                f,
                kernel_layout));
      };
      if constexpr (config::full_waves && decltype(vec)::value == 2) {
        // Whole vector waves keep the original kernel and its payload
        // scheduling.
        if (numel % (2 * int64_t(at::xpu::getDeviceMaxWorkItems())) != 0) {
          return submit(std::true_type{});
        }
      }
      submit(std::false_type{});
    };
    auto launch_vector = [&](auto vec) {
      constexpr int vec_size = decltype(vec)::value;
      constexpr int vec_bytes = vec_size * config::dense_bytes;
      if (reinterpret_cast<uintptr_t>(data[0]) %
                  (vec_size * config::output_bytes) !=
              0 ||
          reinterpret_cast<uintptr_t>(data[dense]) % vec_bytes != 0) {
        return false;
      }
      if constexpr (single_element) {
        launch_layout(
            vec, SingleElementBroadcast<broadcast_arg>{uint32_t(numel)});
        return true;
      } else if constexpr (config::vectorize_rows) {
        if (iter.shape()[0] % vec_size != 0) {
          return false;
        }
        for (int dim = 1; dim < ndim; ++dim) {
          if (iter.strides(dense)[dim] % vec_bytes != 0) {
            return false;
          }
        }
        launch_layout(
            vec,
            InnerBroadcast{
                uint32_t(numel), make_input_offset_calculator<2>(iter)});
        return true;
      }
      return false;
    };
    if constexpr (config::low_precision) {
      const int64_t wave_items = at::xpu::getDeviceMaxWorkItems();
      int64_t vec8_items = wave_items;
      if constexpr (single_element) {
        // In-place scalar broadcasts can pack wider once the grid can offer
        // one work-group per Xe core.
        if (data[0] == data[dense]) {
          vec8_items = launch_config::broadcast_wide_work_group_size *
              int64_t(at::xpu::getDeviceXeCoreCount());
        }
      }
      constexpr int vec4_waves = single_element ? 2 : 1;
      return (numel / 8 >= vec8_items &&
              launch_vector(std::integral_constant<int, 8>{})) ||
          (numel / 4 >= vec4_waves * wave_items &&
           launch_vector(std::integral_constant<int, 4>{})) ||
          launch_vector(std::integral_constant<int, 2>{});
    } else if constexpr (!single_element && config::float_storage) {
      const int64_t wave_items = at::xpu::getDeviceMaxWorkItems();
      return (numel / 4 >= 2 * wave_items &&
              launch_vector(std::integral_constant<int, 4>{})) ||
          (numel / 2 >= wave_items &&
           launch_vector(std::integral_constant<int, 2>{})) ||
          launch_vector(std::integral_constant<int, 1>{});
    } else {
      if constexpr (config::wide_vec == config::narrow_vec) {
        return launch_vector(std::integral_constant<int, config::wide_vec>{});
      } else if (config::wide(numel)) {
        const bool launched =
            launch_vector(std::integral_constant<int, config::wide_vec>{});
        if (single_element || launched) {
          return launched;
        }
      }
      return launch_vector(std::integral_constant<int, config::narrow_vec>{});
    }
  };
  auto launch = [&](auto single) {
    return launch_arg(std::integral_constant<int, 0>{}, single) ||
        launch_arg(std::integral_constant<int, 1>{}, single);
  };
  return ndim == 1 ? launch(std::true_type{}) : launch(std::false_type{});
}

// An inner broadcast may need values from two adjacent rows.
template <int vec_size, typename input_t, typename in_calc_t>
static inline auto load_strided_unary_input(
    bool broadcast,
    const input_t* input,
    uint32_t first,
    at::detail::IntDivider<uint32_t> div0,
    uint32_t base,
    uint32_t size0,
    const in_calc_t& ic) {
  if (!broadcast) {
    return memory::load_vector<vec_size>(input + first, 0);
  }
  const uint32_t pos0 = div0.divmod(base).mod;
  const uint32_t second =
      pos0 + vec_size > size0 ? ic.get(base - pos0 + size0)[0] : first;
  const input_t a = c10::load(input + first);
  const input_t b = c10::load(input + second);
  memory::aligned_vector<input_t, vec_size> values;
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    values.val[i] = pos0 + i < size0 ? a : b;
  }
  return values;
}

template <int vec_size, typename func_t, typename... inputs_t>
static inline void apply_strided_vector(
    const func_t& f,
    typename function_traits<func_t>::result_type* output,
    int64_t index,
    const inputs_t&... inputs) {
  using output_t = typename function_traits<func_t>::result_type;
  memory::aligned_vector<output_t, vec_size> results;
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    results.val[i] = f(inputs.val[i]...);
  }
  reinterpret_cast<memory::aligned_vector<output_t, vec_size>*>(output)[index] =
      results;
}

template <int vec_size, typename func_t, typename array_t, typename in_calc_t>
static inline void launch_strided_unary(
    int numel,
    const func_t& f,
    const array_t& data,
    const in_calc_t& ic,
    bool broadcast,
    uint32_t size0) {
  using traits = function_traits<func_t>;
  using input_t = std::decay_t<typename traits::template arg<0>::type>;
  using output_t = typename traits::result_type;
  constexpr int work_group_size = strided_work_group_size;
  const int vectors = numel / vec_size;
  const int64_t items = vectors + (numel % vec_size != 0);
  const auto groups = ceil_div<int64_t>(items, work_group_size);
  const auto div0 = at::detail::IntDivider<uint32_t>(size0);
  const auto* input0 = reinterpret_cast<const input_t*>(data[1]);
  auto* output = reinterpret_cast<output_t*>(data[0]);
  using kernel_name =
      StridedElementwiseKernel<vec_size, true, func_t, in_calc_t>;
  const sycl::nd_range<1> range(groups * work_group_size, work_group_size);
  getCurrentSYCLQueue().submit([&](sycl::handler& handler) {
    handler.parallel_for<kernel_name>(
        range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
          const int64_t index = int64_t(item.get_group(0)) * work_group_size +
              item.get_local_id(0);
          if (index < vectors) {
            const uint32_t base = uint32_t(index) * vec_size;
            const auto offsets = ic.get(base);
            const auto a = load_strided_unary_input<vec_size>(
                broadcast, input0, offsets[0], div0, base, size0, ic);
            apply_strided_vector<vec_size>(f, output, index, a);
          } else if (index == vectors) {
            for (int i = vectors * vec_size; i < numel; ++i) {
              output[i] = f(c10::load(input0 + ic.get(i)[0]));
            }
          }
        });
  });
}

template <int vec_size, typename func_t, typename array_t, typename in_calc_t>
static inline void launch_strided_binary(
    int numel,
    const func_t& f,
    const array_t& data,
    const in_calc_t& ic) {
  using traits = function_traits<func_t>;
  using input0_t = std::decay_t<typename traits::template arg<0>::type>;
  using input1_t = std::decay_t<typename traits::template arg<1>::type>;
  using output_t = typename traits::result_type;
  constexpr int work_group_size = strided_work_group_size;
  const int vectors = numel / vec_size;
  const auto groups = ceil_div<int64_t>(vectors, work_group_size);
  const auto* input0 = reinterpret_cast<const input0_t*>(data[1]);
  const auto* input1 = reinterpret_cast<const input1_t*>(data[2]);
  auto* output = reinterpret_cast<output_t*>(data[0]);
  using kernel_name =
      StridedElementwiseKernel<vec_size, false, func_t, in_calc_t>;
  const sycl::nd_range<1> range(groups * work_group_size, work_group_size);
  getCurrentSYCLQueue().submit([&](sycl::handler& handler) {
    handler.parallel_for<kernel_name>(range, [=](sycl::nd_item<1> item) {
      const int64_t index =
          int64_t(item.get_group(0)) * work_group_size + item.get_local_id(0);
      if (index < vectors) {
        const uint32_t base = uint32_t(index) * vec_size;
        const auto offsets = strided_binary_offsets(ic, base);
        const auto a = memory::load_vector<vec_size>(input0 + offsets[0], 0);
        const auto b = memory::load_vector<vec_size>(input1 + offsets[1], 0);
        apply_strided_vector<vec_size>(f, output, index, a, b);
      }
    });
  });
}

static inline bool is_aligned_strided_row(
    const TensorIteratorBase& iter,
    int input,
    int vec_size,
    bool allow_1d_tail = false) {
  const auto element_size = iter.element_size(input);
  const auto strides = iter.strides(input);
  const auto vector_bytes = element_size * vec_size;
  const bool needs_full_rows = iter.ndim() > 1 || !allow_1d_tail;
  if (vector_bytes > 16 || strides[0] != element_size ||
      (needs_full_rows && iter.shape()[0] % vec_size != 0) ||
      reinterpret_cast<uintptr_t>(iter.data_ptr(input)) % vector_bytes != 0) {
    return false;
  }
  for (int dim = 1; dim < iter.ndim(); ++dim) {
    if (strides[dim] % vector_bytes != 0) {
      return false;
    }
  }
  return true;
}

// Pack aligned unary rows and inner broadcasts.
template <typename func_t, typename array_t>
static inline bool try_launch_strided_unary(
    TensorIteratorBase& iter,
    const func_t& f,
    const array_t& data,
    bool dense_output) {
  using traits = function_traits<func_t>;
  static_assert(traits::arity == 1);
  using output_t = typename traits::result_type;
  constexpr int vec_size = std::min<int>(8, 16 / sizeof(output_t));
  if constexpr (vec_size < 2) {
    return false;
  } else {
    if (iter.ndim() < 1 || !dense_output ||
        memory::can_vectorize_up_to<output_t>(data[0]) < vec_size) {
      return false;
    }
    const uint32_t size0 = iter.shape()[0];
    const bool broadcast = iter.strides(1)[0] == 0;
    if (broadcast ? size0 < vec_size
                  : !is_aligned_strided_row(iter, 1, vec_size)) {
      return false;
    }
    auto ic = make_input_offset_calculator<1>(iter);
    launch_strided_unary<vec_size>(iter.numel(), f, data, ic, broadcast, size0);
    return true;
  }
}

// Binary vectors stay within one aligned row.
template <typename func_t, typename array_t>
static inline bool try_launch_strided_binary(
    TensorIteratorBase& iter,
    const func_t& f,
    const array_t& data,
    bool dense_output) {
  using traits = function_traits<func_t>;
  static_assert(traits::arity == 2);
  using output_t = typename traits::result_type;
  constexpr int max_bytes = max_scalar_size<func_t>();
  if constexpr (max_bytes > 4) {
    return false;
  } else {
    if (iter.ndim() < 2 || !dense_output) {
      return false;
    }
    const auto shape = iter.shape();
    for (int input = 1; input <= 2; ++input) {
      const auto strides = iter.strides(input);
      const auto begin = reinterpret_cast<uintptr_t>(data[input]);
      auto end = begin + iter.element_size(input);
      bool broadcast = false;
      for (int dim = 0; dim < iter.ndim(); ++dim) {
        broadcast |= shape[dim] > 1 && strides[dim] == 0;
        end += (shape[dim] - 1) * strides[dim];
      }
      if (broadcast && overlaps_dense_output(iter, data[0], begin, end)) {
        return false;
      }
    }
    constexpr int max_vec_size = std::min(8, 16 / max_bytes);
    const auto numel = iter.numel();
    // Use the largest supported subgroup to size a wave of vectors. Require
    // two waves for packing four 32-bit elements per work-item. Keep wider
    // packed accesses for sub-dword elements with one wave.
    const int64_t wave_items = at::xpu::getDeviceMaxWorkItems();
    for (int vec_size = max_vec_size; vec_size > 1; vec_size /= 2) {
      const int waves = max_bytes == 4 && vec_size == 4 ? 2 : 1;
      if (numel / vec_size < waves * wave_items ||
          memory::can_vectorize_up_to<output_t>(data[0]) < vec_size ||
          shape[0] % vec_size != 0 ||
          !is_aligned_strided_row(iter, 1, vec_size) ||
          !is_aligned_strided_row(iter, 2, vec_size)) {
        continue;
      }
      auto launch = [&](auto width) {
        constexpr int width_v = decltype(width)::value;
        if constexpr (width_v <= max_vec_size) {
          if (vec_size == width_v) {
            auto ic = make_input_offset_calculator<2>(iter);
            launch_strided_binary<width_v>(numel, f, data, ic);
            return true;
          }
        }
        return false;
      };
      return launch(std::integral_constant<int, 8>{}) ||
          launch(std::integral_constant<int, 4>{}) ||
          launch(std::integral_constant<int, 2>{});
    }
    return false;
  }
}

// Packed-strided unary dispatch is independent of contiguous launch geometry.
// Captured CPU-scalar functors are unary here too.
template <
    typename func_t,
    bool enable_broadcast_vec = true,
    bool packed_strided_unary = false>
void gpu_kernel_impl_nocast(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);
  TORCH_INTERNAL_ASSERT(!needs_dynamic_casting<func_t>::check(iter));

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();
  bool contiguous = iter.is_contiguous();
  bool latency_case = numel <= static_cast<int64_t>(
                                   at::xpu::getDeviceMaxSubGroupSize() *
                                   at::xpu::getDeviceEUCountPerXeCore()) *
          4; /* on tuning for different data types */

  // Prefer contiguous indexing and aligned-row vectors. Packed strided paths
  // handle remaining legal layouts.
  int vec_size;
  if (contiguous) {
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    vec_size = memory::can_vectorize_up_to<func_t>(data);
    launch_vectorized_kernel(numel, f, data, input_calc, vec_size);
    return;
  } else {
    const bool dense_output = (packed_strided_unary || traits::arity == 2) &&
        has_dense_output(iter, sizeof(arg0_t));
    // Keep the legacy row-vector restrictions for split iterators. Packed
    // strided paths validate their own geometry and also serve split iterators.
    if constexpr (traits::arity == 2) {
      if (try_launch_broadcast_kernel(iter, f, data, dense_output)) {
        return;
      }
    }
    if constexpr (enable_broadcast_vec) {
      if (!latency_case &&
          can_vectorize_for_non_contigouous<func_t>(iter, data, vec_size)) {
        auto input_calc = make_input_offset_calculator<traits::arity>(iter);
        launch_vectorized_kernel(numel, f, data, input_calc, vec_size);
        return;
      }
    }
    if constexpr (traits::arity == 2) {
      if (!latency_case &&
          try_launch_strided_binary(iter, f, data, dense_output)) {
        return;
      }
    } else if constexpr (packed_strided_unary) {
      if (try_launch_strided_unary(iter, f, data, dense_output)) {
        return;
      }
    }
  }

  auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
  launch_legacy_global_range_kernel(
      numel,
      LegacyKernelScalarFunctor<
          arg0_t,
          ntensors,
          decltype(offset_calc),
          func_t>(data, offset_calc, f));
}

template <
    typename low_t,
    int float_mask,
    typename func_t,
    typename calc_t,
    bool scalar_broadcast = false>
struct MixedDtypeVectorizedKernel {
  using traits = function_traits<func_t>;
  using output_t = typename traits::result_type;
  static constexpr int arity = traits::arity;
  // Only the compact layout broadcasts every FP32 input.
  static constexpr int vec_size =
      scalar_broadcast && sizeof(output_t) <= 2 ? 8 : 4;

  template <size_t i>
  using storage_t =
      std::conditional_t<(float_mask & (1 << i)) != 0, float, low_t>;

  template <size_t i>
  using arg_t = typename traits::template arg<i>::type;

  template <size_t i>
  static arg_t<i> convert_input(storage_t<i> value) {
    if constexpr (
        std::is_same_v<storage_t<i>, float> && std::is_same_v<arg_t<i>, Half>) {
      // Preserve the Half bits when the functor widens to FP32 for math.
      return Half(
          c10::detail::fp16_ieee_from_fp32_value(value), Half::from_bits());
    }
    return c10::convert<arg_t<i>>(value);
  }

  template <size_t i>
  bool is_broadcast() const {
    if constexpr (scalar_broadcast) {
      return (float_mask & (1 << i)) != 0;
    }
    return (broadcast_mask & (1 << i)) != 0;
  }

  template <size_t i>
  const storage_t<i>* input_ptr(uint32_t offset) const {
    if constexpr (is_trivial_offset_calculator_v<calc_t>) {
      offset = is_broadcast<i>() ? 0 : offset;
    }
    return reinterpret_cast<const storage_t<i>*>(data[i + 1]) + offset;
  }

  template <size_t i>
  auto load_vector(uint32_t offset) const {
    const auto* input = input_ptr<i>(offset);
    memory::aligned_vector<arg_t<i>, vec_size> values;
    if (is_broadcast<i>()) {
      const auto value = convert_input<i>(c10::load(input));
#pragma unroll
      for (int j = 0; j < vec_size; ++j) {
        values.val[j] = value;
      }
    } else {
      const auto loaded = memory::load_vector<vec_size>(input, 0);
#pragma unroll
      for (int j = 0; j < vec_size; ++j) {
        values.val[j] = convert_input<i>(loaded.val[j]);
      }
    }
    return values;
  }

  template <size_t... i>
  void apply_vector(uint32_t vector_index, std::index_sequence<i...>) const {
    const auto offsets = calc.get(vector_index * vec_size);
    auto* output = reinterpret_cast<output_t*>(data[0]);
    apply_strided_vector<vec_size>(
        f, output, vector_index, load_vector<i>(offsets[i])...);
  }

  template <size_t... i>
  void apply_scalar(uint32_t index, std::index_sequence<i...>) const {
    auto* output = reinterpret_cast<output_t*>(data[0]);
    output[index] = f(convert_input<i>(c10::load(input_ptr<i>(index)))...);
  }

  [[sycl::reqd_sub_group_size(launch_config::mixed_dtype_sub_group_size)]] void
  operator()(sycl::nd_item<1> item) const {
    constexpr auto indices = std::make_index_sequence<arity>{};
    const uint32_t vector_index = static_cast<uint32_t>(item.get_group(0)) *
            launch_config::broadcast_work_group_size +
        static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t full_vectors = numel / vec_size;
    if (vector_index < full_vectors) {
      apply_vector(vector_index, indices);
    } else if constexpr (is_trivial_offset_calculator_v<calc_t>) {
      if (vector_index == full_vectors) {
        for (uint32_t index = vector_index * vec_size; index < numel; ++index) {
          apply_scalar(index, indices);
        }
      }
    }
  }

  at::detail::Array<char*, arity + 1> data;
  calc_t calc;
  func_t f;
  int broadcast_mask;
  uint32_t numel;
};

template <typename low_t, int mask = 1, typename func_t, typename calc_t>
static inline void launch_mixed_dtype_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    int float_mask,
    int broadcast_mask,
    bool wide_aligned,
    const calc_t& calc) {
  constexpr int arity = function_traits<func_t>::arity;
  if (float_mask == mask) {
    at::detail::Array<char*, arity + 1> data;
    for (int i = 0; i <= arity; ++i) {
      data[i] = static_cast<char*>(iter.data_ptr(i));
    }
    const auto numel = static_cast<uint32_t>(iter.numel());
    auto submit = [&](auto kernel) {
      constexpr int work_group_size = launch_config::broadcast_work_group_size;
      constexpr int vec_size = decltype(kernel)::vec_size;
      const auto groups = ceil_div<int64_t>(numel, work_group_size * vec_size);
      sycl_kernel_submit(
          groups * work_group_size,
          work_group_size,
          getCurrentSYCLQueue(),
          kernel);
    };
    if constexpr (is_trivial_offset_calculator_v<calc_t>) {
      using scalar_kernel_t =
          MixedDtypeVectorizedKernel<low_t, mask, func_t, calc_t, true>;
      if (broadcast_mask == mask &&
          (scalar_kernel_t::vec_size == 4 || wide_aligned)) {
        submit(scalar_kernel_t{data, calc, f, broadcast_mask, numel});
        return;
      }
    }
    using kernel_t = MixedDtypeVectorizedKernel<low_t, mask, func_t, calc_t>;
    submit(kernel_t{data, calc, f, broadcast_mask, numel});
  } else if constexpr (mask + 1 < (1 << arity) - 1) {
    launch_mixed_dtype_kernel<low_t, mask + 1>(
        iter, f, float_mask, broadcast_mask, wide_aligned, calc);
  }
}

template <typename args_t>
struct MixedDtypeArgsSupported;

template <typename... args_t>
struct MixedDtypeArgsSupported<std::tuple<args_t...>>
    : std::bool_constant<(
          (std::is_same_v<args_t, float> || std::is_same_v<args_t, Half> ||
           std::is_same_v<args_t, BFloat16>)&&...)> {};

// Load FP32 plus one low-precision storage type without dynamic dtype switches,
// preserving each functor argument's conversion. Vectors require dense output
// and aligned unit/zero inner strides. Only 1D layouts can have scalar tails.
template <typename func_t, bool enable_broadcast_vec>
static inline bool try_launch_mixed_dtype_kernel(
    TensorIteratorBase& iter,
    const func_t& f) {
  using traits = function_traits<func_t>;
  using output_t = typename traits::result_type;
  constexpr int arity = traits::arity;
  if constexpr (
      (arity == 2 || arity == 3) &&
      MixedDtypeArgsSupported<typename traits::ArgsTuple>::value &&
      (std::is_same_v<output_t, float> || std::is_same_v<output_t, Half> ||
       std::is_same_v<output_t, BFloat16> || std::is_same_v<output_t, bool>)) {
    if (iter.dtype(0) != c10::CppTypeToScalarType<output_t>::value ||
        iter.numel() < 4 || (!enable_broadcast_vec && !iter.is_contiguous())) {
      return false;
    }
    int float_mask = 0;
    int broadcast_mask = 0;
    ScalarType low_dtype = kFloat;
    for (int i = 0; i < arity; ++i) {
      const auto dtype = iter.dtype(i + 1);
      if (dtype == kFloat) {
        float_mask |= 1 << i;
      } else if (dtype == kHalf || dtype == kBFloat16) {
        if (low_dtype != kFloat && low_dtype != dtype) {
          return false;
        }
        low_dtype = dtype;
      } else {
        return false;
      }
    }
    if (float_mask == 0 || float_mask == (1 << arity) - 1) {
      return false;
    }
    if (!has_dense_output(iter, sizeof(output_t))) {
      return false;
    }
    bool wide_aligned = iter.ndim() == 1 && sizeof(output_t) <= 2;
    for (int i = 0; i <= arity; ++i) {
      if (i != 0) {
        if (broadcast_overlaps_output(iter, iter.data_ptr(0), i, true)) {
          return false;
        }
        if (iter.strides(i)[0] == 0) {
          broadcast_mask |= 1 << (i - 1);
          continue;
        }
      }
      if (!is_aligned_strided_row(iter, i, 4, /*allow_1d_tail=*/true)) {
        return false;
      }
      wide_aligned = wide_aligned &&
          is_aligned_strided_row(iter, i, 8, /*allow_1d_tail=*/true);
    }
    auto launch = [&](const auto& calc) {
      if (low_dtype == kHalf) {
        launch_mixed_dtype_kernel<Half>(
            iter, f, float_mask, broadcast_mask, wide_aligned, calc);
      } else {
        launch_mixed_dtype_kernel<BFloat16>(
            iter, f, float_mask, broadcast_mask, wide_aligned, calc);
      }
    };
    if (iter.ndim() == 1) {
      launch(TrivialOffsetCalculator<arity>{});
    } else {
      launch(make_input_offset_calculator<arity>(iter));
    }
    return true;
  }
  return false;
}

template <
    typename func_t,
    bool enable_broadcast_vec = true,
    bool packed_strided_unary = false>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  if (!needs_dynamic_casting<func_t>::check(iter)) {
    return gpu_kernel_impl_nocast<
        func_t,
        enable_broadcast_vec,
        packed_strided_unary>(iter, f);
  }
  // Try typed mixed-storage vectors before dynamic casting; remaining casts
  // use the FP-only switch when legal, then the generic loader/storer.
  if (try_launch_mixed_dtype_kernel<func_t, enable_broadcast_vec>(iter, f)) {
    return;
  }
  using traits = function_traits<func_t>;
  using arg0_t = typename traits::result_type;
  constexpr int ntensors = traits::arity + 1;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
  TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  // Fast path for mixed fp32/bf16/fp16 elementwise ops using a compact
  // 3-case switch (LoadWithCastFP/StoreWithCastFP) to avoid register
  // spilling from the generic large ScalarType switch (see #4904).
  // fp_result (compile-time) ensures the functor output type is exactly
  // float32 (excluding double, bool, etc);
  // dtype(0)==Float (runtime) guarantees the output pointer is float*
  // as StoreWithCastFP expects.
  constexpr bool fp_result = std::is_same_v<arg0_t, float>;
  bool use_fp_cast = fp_result && (iter.dtype(0) == at::ScalarType::Float);
  if (use_fp_cast) {
    for (int i = 1; i < ntensors; i++) {
      auto dt = iter.dtype(i);
      if (dt != at::ScalarType::Float && dt != at::ScalarType::Half &&
          dt != at::ScalarType::BFloat16) {
        use_fp_cast = false;
        break;
      }
    }
  }

  bool contiguous = iter.is_contiguous();

  if (contiguous) {
    if (use_fp_cast) {
      auto loader = memory::LoadWithCastFP<traits::arity>(iter);
      auto storer = memory::StoreWithCastFP<1>();
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      launch_unrolled_kernel(
          numel,
          f,
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);
    } else {
      auto loader = memory::LoadWithCast<traits::arity>(iter);
      auto storer = memory::StoreWithCast<1>(iter);
      auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
      auto output_offset_calculator = TrivialOffsetCalculator<1>();
      launch_unrolled_kernel(
          numel,
          f,
          data,
          input_offset_calculator,
          output_offset_calculator,
          loader,
          storer);
    }
  } else {
    at::detail::Array<ScalarType, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
      dtypes[i] = iter.dtype(i);
    }
    auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);
    constexpr int unroll_factor = sizeof(arg0_t) > 4 ? 2 : 4;
    launch_legacy_group_range_kernel<unroll_factor>(
        numel,
        LegacyKernelWithCastScalarFunctor<
            arg0_t,
            ntensors,
            decltype(offset_calc),
            func_t>(data, dtypes, offset_calc, f));
  }
}

template <typename func_t>
void gpu_kernel_nocast(TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
        iter.device(arg).is_xpu(),
        "argument ",
        arg,
        ": expected an XPU device but found ",
        iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_nocast(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl_nocast(iter, f);
}

// Unary functors, including captured CPU scalars, enable packed-strided
// dispatch. Contiguous work uses the common vectorized launcher.
template <typename func_t, bool enable_broadcast_vec = true>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(
        iter.device(arg).is_xpu(),
        "argument ",
        arg,
        ": expected an XPU device but found ",
        iter.device(arg));
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      // Keep the legacy row-vector and mixed-storage strided restrictions for
      // split iterators; iterator-local broadcast/packed-strided paths remain
      // eligible and validate their own strides and pointers.
      gpu_kernel<func_t, false>(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl<
      func_t,
      enable_broadcast_vec,
      function_traits<func_t>::arity == 1>(iter, f);
}

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct AUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  return_t operator()(arg2_t b) const {
    return f(a, b);
  }
  AUnaryFunctor(func_t f_, opmath_arg1_t a_) : f(f_), a(a_) {}

 private:
  [[no_unique_address]] func_t f;
  opmath_arg1_t a;
};

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor {
  using traits = function_traits<func_t>;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  return_t operator()(arg1_t a) const {
    return f(a, b);
  }
  BUnaryFunctor(func_t f_, opmath_arg2_t b_) : f(f_), b(b_) {}

 private:
  [[no_unique_address]] func_t f;
  opmath_arg2_t b;
};

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
  return_t operator()(arg1_t a, arg2_t b) const {
    return f(a, b);
  }
  BinaryFunctor(func_t f_) : f(f_) {}

 private:
  func_t f;
};

template <typename func_t>
bool opmath_gpu_kernel_with_mixed_types(
    TensorIteratorBase& iter,
    const func_t& f) {
  if (iter.dtype(0) != kFloat || iter.dtype(1) == iter.dtype(2) ||
      !iter.device(1).is_xpu() || !iter.device(2).is_xpu()) {
    return false;
  }

  // Keep the casting fallback when a broadcast input overlaps the output.
  if (iter.numel() > 1 && !iter.is_contiguous() &&
      has_dense_output(iter, sizeof(float))) {
    const auto* output = iter.data_ptr(0);
    if (broadcast_overlaps_output(iter, output, 1, /*broadcast_only=*/true) ||
        broadcast_overlaps_output(iter, output, 2, /*broadcast_only=*/true)) {
      return false;
    }
  }

  // Fix the storage types on the host so the existing nocast path can
  // vectorize.
  if (iter.dtype(1) == kFloat) {
    if (iter.dtype(2) == kHalf) {
      gpu_kernel(iter, BinaryFunctor<float, Half, float, func_t>(f));
    } else if (iter.dtype(2) == kBFloat16) {
      gpu_kernel(iter, BinaryFunctor<float, BFloat16, float, func_t>(f));
    } else {
      return false;
    }
  } else if (iter.dtype(2) == kFloat) {
    if (iter.dtype(1) == kHalf) {
      gpu_kernel(iter, BinaryFunctor<Half, float, float, func_t>(f));
    } else if (iter.dtype(1) == kBFloat16) {
      gpu_kernel(iter, BinaryFunctor<BFloat16, float, float, func_t>(f));
    } else {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

template <
    typename arg1_t,
    typename arg2_t,
    typename return_t,
    bool enable_mixed_types = true,
    typename func_t>
void opmath_gpu_kernel_binary(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  if constexpr (
      enable_mixed_types && std::is_same_v<arg1_t, float> &&
      std::is_same_v<arg2_t, float> && std::is_same_v<return_t, float> &&
      std::is_same_v<typename traits::template arg<0>::type, float> &&
      std::is_same_v<typename traits::template arg<1>::type, float> &&
      std::is_same_v<typename traits::result_type, float>) {
    if (opmath_gpu_kernel_with_mixed_types(iter, f)) {
      return;
    }
  }
  gpu_kernel(iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
}

template <
    typename arg1_t,
    typename arg2_t = arg1_t,
    typename return_t = arg1_t,
    typename func_t,
    bool enable_mixed_types = true>
void opmath_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg1_t = typename traits::template arg<0>::type;
  using opmath_arg2_t = typename traits::template arg<1>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");

  if (iter.is_cpu_scalar(1)) {
    AUnaryFunctor<arg1_t, arg2_t, return_t, func_t> af(
        f, iter.scalar_value<opmath_arg1_t>(1));
    iter.remove_operand(1);
    const OptionalDeviceGuard device_guard(iter.device(1));
    gpu_kernel(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(
        f, iter.scalar_value<opmath_arg2_t>(2));
    iter.remove_operand(2);
    gpu_kernel(iter, bf);
  } else {
    opmath_gpu_kernel_binary<arg1_t, arg2_t, return_t, enable_mixed_types>(
        iter, f);
  }
}

template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars(
    TensorIteratorBase& iter,
    const func_t& f) {
  // Use symmetric property of the functor to reduce number of kernels,
  // requires f(a, b) == f(b, a)
  TORCH_INTERNAL_ASSERT(iter.ntensors() == 3);

  using traits = function_traits<func_t>;
  using opmath_arg_t = typename traits::template arg<0>::type;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  static_assert(
      std::is_same_v<opmath_arg_t, typename traits::template arg<1>::type>,
      "f is not symmetric");

  OptionalDeviceGuard device_guard;
  opmath_arg_t scalar_val{};

  if (iter.is_cpu_scalar(1)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(1);
    iter.remove_operand(1);

    device_guard.reset_device(iter.device(1));
  } else if (iter.is_cpu_scalar(2)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(2);
    iter.remove_operand(2);
  }

  if (iter.ninputs() == 2) {
    opmath_gpu_kernel_binary<scalar_t, scalar_t, return_t>(iter, f);
  } else {
    AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f, scalar_val);
    gpu_kernel(iter, unary_f);
  }
}

// Legacy variant that assumes that func_t has the correct types
// that we expect to load from memory
template <typename func_t>
void gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  opmath_gpu_kernel_with_scalars<arg1_t, arg2_t, return_t, func_t, false>(
      iter, f);
}

template <typename func_t>
void gpu_kernel_multiple_outputs_impl(
    TensorIteratorBase& iter,
    const func_t& f) {
  using traits = function_traits<func_t>;
  using output_t = typename traits::result_type;
  constexpr int num_outputs = std::tuple_size_v<output_t>;
  constexpr int num_inputs = traits::arity;
  constexpr int ntensors = num_outputs + num_inputs;

  TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
  TORCH_INTERNAL_ASSERT(iter.ntensors() == ntensors);

  at::detail::Array<char*, ntensors> data;
  for (int i = 0; i < ntensors; i++) {
    data[i] = (char*)iter.data_ptr(i);
  }

  int64_t numel = iter.numel();

  if (iter.is_contiguous()) {
    auto input_calc = TrivialOffsetCalculator<num_inputs>();
    auto output_calc = TrivialOffsetCalculator<num_outputs>();
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(
        numel, f, data, input_calc, output_calc);
  } else {
    auto input_calc = make_input_offset_calculator<num_inputs>(iter);
    auto output_calc = make_output_offset_calculator<num_outputs>(iter);
    launch_unrolled_kernel_for_multi_outputs<num_outputs>(
        numel, f, data, input_calc, output_calc);
  }
}

template <typename func_t>
void gpu_kernel_multiple_outputs(TensorIteratorBase& iter, const func_t& f) {
  for (int arg = 0; arg < iter.ntensors(); arg++) {
    TORCH_INTERNAL_ASSERT(iter.device(arg).is_xpu());
  }

  if (iter.numel() == 0) {
    return;
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      gpu_kernel_multiple_outputs(sub_iter, f);
    }
    return;
  }

  gpu_kernel_multiple_outputs_impl(iter, f);
}

} // namespace xpu
} // namespace native
} // namespace at
