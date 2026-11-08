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

template <int vec_size, typename func_t, typename array_t, typename in_calc_t>
struct VectorizedElementwiseKernel {
  void operator()(sycl::nd_item<1> item) const {
    int grpsz = item.get_local_range(0);
    int grpid = item.get_group(0);
    int lid = item.get_local_id(0);
    int group_work_size = vec_size * grpsz;
    int remaining = numel_ - grpid * group_work_size;

    // ic_
    if (remaining < group_work_size) {
      auto oc = TrivialOffsetCalculator<1>();
      auto l = at::native::memory::LoadWithoutCast();
      auto s = at::native::memory::StoreWithoutCast();
      auto policy = at::native::memory::policies::unroll<
          vec_size,
          array_t,
          decltype(ic_),
          decltype(oc),
          at::native::memory::LoadWithoutCast,
          at::native::memory::StoreWithoutCast>(
          data_, remaining, ic_, oc, l, s, lid, grpid, grpsz);
      elementwise_kernel_helper<vec_size>(f_, policy);
    } else {
      auto policy = at::native::memory::policies::
          vectorized<vec_size, array_t, in_calc_t>(
              data_, ic_, lid, grpid, grpsz);
      elementwise_kernel_helper<vec_size>(f_, policy);
    }
  }

  VectorizedElementwiseKernel(
      int numel,
      const func_t f,
      array_t data,
      in_calc_t ic)
      : numel_(numel), f_(f), data_(data), ic_(ic) {}

 private:
  int numel_;
  const func_t f_;
  array_t data_;
  in_calc_t ic_;
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

// Keep launch and indexing geometry in sync within each path. Configurations
// are independent even where their current values coincide.
namespace launch_config {
constexpr int compact_work_group_size = 128;
constexpr int compact_sub_group_size = 32;

constexpr int strided_unary_work_group_size = 128;
constexpr int strided_unary_sub_group_size = 32;

// Only the wide broadcast path fixes the subgroup size.
constexpr int broadcast_work_group_size = 256;
constexpr int broadcast_wide_work_group_size = 128;
constexpr int broadcast_wide_sub_group_size = 16;
} // namespace launch_config

template <int vec_size, typename func_t, bool bounds>
class CompactScalarElementwiseKernel;

// A captured empty functor still occupies a padded by-value kernel argument
// (e.g. 8 x 1 B between two pointers), which can push the payload past the
// inline cross-thread data. Such functors are rebuilt inside the kernel.
template <typename func_t>
constexpr bool is_stateless_functor_v =
    std::is_empty_v<func_t> && std::is_default_constructible_v<func_t>;

template <int vec_size, typename func_t, typename input_t, typename output_t>
static inline void compact_unary_apply(
    const input_t* input,
    const func_t& f,
    output_t* output,
    int index,
    bool vectorized) {
  if (vectorized) {
    auto values = memory::load_vector<vec_size>(input, index);
    memory::aligned_vector<output_t, vec_size> results;
#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      results.val[i] = f(values.val[i]);
    }
    reinterpret_cast<memory::aligned_vector<output_t, vec_size>*>(
        output)[index] = results;
  } else {
    output[index] = f(c10::load(input + index));
  }
}

template <
    typename input_t,
    typename output_t,
    typename func_t,
    typename scalar_arg_t,
    bool scalar_left>
struct ScalarUnaryFunctor {
  output_t operator()(input_t value) const {
    if constexpr (scalar_left) {
      return f(scalar, value);
    } else {
      return f(value, scalar);
    }
  }
  ScalarUnaryFunctor(func_t f_, scalar_arg_t scalar_)
      : scalar(scalar_), f(f_) {}

  scalar_arg_t scalar;
  func_t f;
};

// Pack contiguous unary and captured CPU-scalar work into WG128/SG32 tiles.
// Full-tile specialization omits the bounds argument/check; partial tiles
// retain scalar tails, and vec1 uses the caller's unrolled path.
template <int vec_size, typename func_t, typename array_t>
static inline void launch_compact_unary_kernel_impl(
    int numel,
    const func_t& f,
    array_t data) {
  using traits = function_traits<func_t>;
  static_assert(traits::arity == 1);
  using input_t = typename traits::template arg<0>::type;
  using output_t = typename traits::result_type;
  constexpr int work_group_size = launch_config::compact_work_group_size;
  constexpr int group_work_size = work_group_size * vec_size;
  const auto* input = reinterpret_cast<const input_t*>(data[1]);
  auto* output = reinterpret_cast<output_t*>(data[0]);
  auto compute = [=](int index, bool vectorized) {
    // `f` is named only in the branch that is kept, so it is captured only
    // when the functor has state. Captures are laid out in order of first
    // use: `input` comes first so its pointer stays in the inline payload.
    if constexpr (is_stateless_functor_v<func_t>) {
      compact_unary_apply<vec_size>(input, func_t{}, output, index, vectorized);
    } else {
      compact_unary_apply<vec_size>(input, f, output, index, vectorized);
    }
  };
  const size_t groups = ceil_div<int64_t>(numel, group_work_size);
  const sycl::nd_range<1> range(groups * work_group_size, work_group_size);
  auto& queue = getCurrentSYCLQueue();
  if (numel % group_work_size == 0) {
    // Full tiles omit the bounds argument from the cross-thread payload.
    queue.submit([&](sycl::handler& handler) {
      handler.parallel_for<
          CompactScalarElementwiseKernel<vec_size, func_t, false>>(
          range,
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(
              launch_config::compact_sub_group_size)]] {
            const int index = int(item.get_group(0)) * work_group_size +
                int(item.get_local_id(0));
            compute(index, true);
          });
    });
  } else {
    queue.submit([&](sycl::handler& handler) {
      handler
          .parallel_for<CompactScalarElementwiseKernel<vec_size, func_t, true>>(
              range,
              [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(
                  launch_config::compact_sub_group_size)]] {
                const int group = int(item.get_group(0));
                const int local = int(item.get_local_id(0));
                const int index = group * work_group_size + local;
                if (index < numel / vec_size) {
                  compute(index, true);
                }
                if (group == numel / group_work_size &&
                    local < numel % vec_size) {
                  compute((numel / vec_size) * vec_size + local, false);
                }
              });
    });
  }
}

template <int vec_size, typename func_t, typename array_t>
static inline void launch_compact_unary_kernel(
    int numel,
    const func_t& f,
    array_t data) {
  launch_compact_unary_kernel_impl<vec_size>(numel, f, data);
}

template <int vec_size, typename func_t, typename in_calc_t>
class StridedUnaryElementwiseKernel;

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

// How a work-item reads the inputs of its vec_size consecutive outputs. The
// mode is uniform across the launch, so selecting it at runtime does not
// diverge and keeps one kernel per functor.
enum class StridedInputMode : int {
  // Innermost input dim is unit-stride and every row is vector-aligned.
  kRowVector,
  // Innermost input dim has stride 0: at most two distinct inputs per vector.
  kBroadcastInner,
  // Otherwise: gather with stride s0; at most one row wrap per vector.
  kStridedInner,
};

// For a unary functor (including tensor-scalar ops whose CPU scalar was
// captured) whose output is dense in iteration order while its input is
// strided or broadcast. The innermost size must be >= vec_size unless rows
// are vector-aligned, so one offset calculation covers the vector and a
// second is needed only where it wraps into the next row.
template <typename func_t, typename array_t>
static inline bool try_launch_strided_input_unary_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    array_t data) {
  using traits = function_traits<func_t>;
  static_assert(traits::arity == 1);
  using input_t = std::decay_t<typename traits::template arg<0>::type>;
  using output_t = typename traits::result_type;
  constexpr int vec_size = std::min<int>(8, 16 / sizeof(output_t));
  if constexpr (vec_size < 2) {
    return false;
  } else {
    if (iter.ndim() == 0) {
      return false;
    }
    const auto shape = iter.shape();
    if (!has_dense_output(iter, sizeof(output_t))) {
      return false;
    }
    if (memory::can_vectorize_up_to<output_t>(data[0]) < vec_size) {
      return false;
    }

    const auto in_strides = iter.strides(1);
    const uint32_t size0 = shape[0];
    const uint32_t stride0 = in_strides[0] / int64_t(sizeof(input_t));
    StridedInputMode mode;
    if (in_strides[0] == 0) {
      mode = StridedInputMode::kBroadcastInner;
    } else {
      mode = StridedInputMode::kStridedInner;
    }
    if (sizeof(input_t) * vec_size <= 16 &&
        in_strides[0] == int64_t(sizeof(input_t)) && size0 % vec_size == 0 &&
        reinterpret_cast<uint64_t>(data[1]) % (sizeof(input_t) * vec_size) ==
            0) {
      mode = StridedInputMode::kRowVector;
      for (int dim = 1; dim < iter.ndim(); dim++) {
        if (in_strides[dim] % int64_t(sizeof(input_t) * vec_size) != 0) {
          mode = StridedInputMode::kStridedInner;
        }
      }
    }
    if (mode != StridedInputMode::kRowVector && size0 < vec_size) {
      return false;
    }

    auto ic = make_input_offset_calculator<1>(iter);
    using in_calc_t = decltype(ic);
    const auto div0 = at::detail::IntDivider<uint32_t>(size0);
    constexpr int work_group_size =
        launch_config::strided_unary_work_group_size;
    const int numel = iter.numel();
    const int vec_numel = numel / vec_size;
    const int64_t items = vec_numel + (numel % vec_size != 0);
    const int64_t groups = ceil_div<int64_t>(items, work_group_size);
    const auto* input = reinterpret_cast<const input_t*>(data[1]);
    auto* output = reinterpret_cast<output_t*>(data[0]);
    getCurrentSYCLQueue().submit([&](sycl::handler& handler) {
      // SG32 improved measured throughput for this strided unary path.
      handler.parallel_for<
          StridedUnaryElementwiseKernel<vec_size, func_t, in_calc_t>>(
          sycl::nd_range<1>(groups * work_group_size, work_group_size),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(
              launch_config::strided_unary_sub_group_size)]] {
            const int64_t index = int64_t(item.get_group(0)) * work_group_size +
                item.get_local_id(0);
            if (index < vec_numel) {
              const uint32_t base = uint32_t(index) * vec_size;
              const uint32_t first = ic.get(base)[0];
              // Load modes share the same straight-line loop over the functor.
              memory::aligned_vector<input_t, vec_size> values;
              if (mode == StridedInputMode::kRowVector) {
                // The shared loader also normalizes noncanonical bool bytes.
                values = memory::load_vector<vec_size>(input + first, 0);
              } else {
                const uint32_t pos0 = div0.divmod(base).mod;
                // Offset of the next innermost row, needed only on a wrap.
                const uint32_t second = pos0 + vec_size > size0
                    ? ic.get(base - pos0 + size0)[0]
                    : first;
                if (mode == StridedInputMode::kBroadcastInner) {
                  const input_t a = c10::load(input + first);
                  const input_t b = c10::load(input + second);
#pragma unroll
                  for (int i = 0; i < vec_size; ++i) {
                    values.val[i] = pos0 + i < size0 ? a : b;
                  }
                } else {
#pragma unroll
                  for (int i = 0; i < vec_size; ++i) {
                    const uint32_t offset = pos0 + i < size0
                        ? first + i * stride0
                        : second + (pos0 + i - size0) * stride0;
                    values.val[i] = c10::load(input + offset);
                  }
                }
              }
              memory::aligned_vector<output_t, vec_size> results;
#pragma unroll
              for (int i = 0; i < vec_size; ++i) {
                results.val[i] = f(values.val[i]);
              }
              reinterpret_cast<memory::aligned_vector<output_t, vec_size>*>(
                  output)[index] = results;
            } else if (index == vec_numel) {
              for (int i = vec_numel * vec_size; i < numel; ++i) {
                output[i] = f(c10::load(input + ic.get(i)[0]));
              }
            }
          });
    });
    return true;
  }
}

template <
    bool compact_unary = false,
    typename func_t,
    typename array_t,
    typename in_calc_t>
static inline void launch_vectorized_kernel(
    int64_t N,
    const func_t& f,
    array_t data,
    in_calc_t input_calc,
    int vec_size) {
  constexpr auto max_scalar_bytes = max_scalar_size<func_t>();
  TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
  using traits = function_traits<func_t>;
  int64_t wg_sz = at::xpu::getDeviceMaxSubGroupSize() *
      at::xpu::getDeviceEUCountPerXeCore();

#define VEC_KER(vec_size)                                                      \
  {                                                                            \
    TORCH_CHECK(max_scalar_bytes* vec_size <= 16);                             \
    if constexpr (max_scalar_bytes * vec_size <= 16) {                         \
      if constexpr (compact_unary) {                                           \
        launch_compact_unary_kernel<vec_size>(N, f, data);                     \
      } else {                                                                 \
        auto ker =                                                             \
            VectorizedElementwiseKernel<vec_size, func_t, array_t, in_calc_t>( \
                N, f, data, input_calc);                                       \
        int64_t num_wg = ceil_div<int64_t>(N, wg_sz * vec_size);               \
        sycl_kernel_submit(wg_sz* num_wg, wg_sz, getCurrentSYCLQueue(), ker);  \
      }                                                                        \
    }                                                                          \
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

// Functors whose arithmetic costs little next to their memory traffic (add,
// mul, ...) may declare `static constexpr bool prefer_wide_vectors = true;`,
// which selects wide work-items where the launch depends on the functor cost.
// The value may depend on the functor's template arguments, e.g. its dtype.
template <typename func_t, typename = void>
struct prefers_wide_vectors : std::false_type {};

template <typename func_t>
struct prefers_wide_vectors<
    func_t,
    std::void_t<decltype(func_t::prefer_wide_vectors)>>
    : std::bool_constant<func_t::prefer_wide_vectors> {};

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

// Otherwise each innermost row of the broadcast input is a single element.
struct InnerBroadcast {
  OffsetCalculator<2>::offset_type get(uint32_t first) const {
    // In two dimensions the quotient is already the outer coordinate;
    // avoid decomposing it again in the generic dimension loop.
    if (calc.dims == 2) {
      const auto divmod = calc.sizes_[0].divmod(first);
      OffsetCalculator<2>::offset_type offsets;
#pragma unroll
      for (int arg = 0; arg < 2; ++arg) {
        offsets[arg] = divmod.mod * calc.strides_[0][arg] +
            divmod.div * calc.strides_[1][arg];
      }
      return offsets;
    }
    return calc.get(first);
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
// stay in the inline kernel payload. Functors preferring wide vectors use the
// SIMD16 geometry with 16 bytes per work-item and no bounds checks for full
// tiles; the others leave the SIMD width to the compiler.
// Work-items past the vectors take one element each: the
// numel % vec_size tail, or with full_waves every element after the whole
// waves of two-element vectors, so a partial last wave of a compute-heavy
// functor holds single elements (see BroadcastLaunchConfig).
template <
    int vec_size,
    int broadcast_arg,
    bool prefer_wide,
    bool bounds,
    typename func_t,
    typename layout_t,
    bool full_waves = false>
struct BroadcastElementwiseKernel {
  static_assert(!full_waves || (bounds && vec_size == 2));
  static constexpr int work_group_size = prefer_wide
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

  // A constant group size keeps the index free of payload loads. With
  // full_waves, layout_.numel holds the number of vectors instead of
  // elements, and the launch has exactly one work-item per later element.
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t index = uint32_t(item.get_group(0)) * work_group_size +
        uint32_t(item.get_local_id(0));
    const uint32_t vectors =
        full_waves ? layout_.numel : layout_.numel / vec_size;
    if (!bounds || index < vectors) {
      const auto offsets = layout_.get(index * vec_size);
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
    } else if (
        bounds && (full_waves || index < vectors + layout_.numel % vec_size)) {
      const uint32_t k = vectors * (vec_size - 1) + index;
      const auto offsets = layout_.get(k);
      output_[k] = apply(
          c10::load(dense_ + offsets[1 - broadcast_arg]),
          c10::load(broadcast_ + offsets[broadcast_arg]));
    }
  }

  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    if constexpr (prefer_wide) {
      return sycl::ext::oneapi::experimental::properties{
          sycl::ext::oneapi::experimental::sub_group_size<
              launch_config::broadcast_wide_sub_group_size>};
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
template <typename func_t, int broadcast_arg>
struct BroadcastLaunchConfig {
  using traits = function_traits<func_t>;
  static constexpr bool prefer_wide = prefers_wide_vectors<func_t>::value;
  static constexpr int dense_bytes =
      sizeof(typename traits::template arg<1 - broadcast_arg>::type);
  static constexpr int output_bytes = sizeof(typename traits::result_type);
  static constexpr int min_bytes = std::min(dense_bytes, output_bytes);
  static constexpr int max_bytes = std::max(dense_bytes, output_bytes);
  // Functors preferring wide vectors take 16 bytes of the widest streamed
  // operand per work-item. Otherwise compute-heavy functors (pow, fmod, ...)
  // need the thread supply more than wide accesses: a work-item takes 8 bytes
  // only while that still gives every hardware thread work, twice over when 4
  // bytes already hold several elements, else 4 bytes. The narrowest operand,
  // e.g. a bool output, keeps at least 4 bytes per work-item.
  static constexpr int narrow_vec = std::max(1, 4 / min_bytes);
  static constexpr int wide_vec =
      std::max((prefer_wide ? 16 : 8) / max_bytes, narrow_vec);
  static constexpr int waves = narrow_vec > 1 ? 2 : 1;
  // A wide launch of a functor that is not marked cheap rarely fills its
  // last wave of two-element vectors. Its work-items run as long as the
  // functor's arithmetic, so that wave would cost as much as a full one;
  // the elements after the whole waves rather take one work-item each.
  static constexpr bool full_waves =
      !prefer_wide && narrow_vec == 1 && wide_vec == 2;
  // Broadcast rows gain from vectors only when a streamed operand is narrower
  // than 8 bytes; otherwise the generic kernel already accesses 8 bytes per
  // work-item, and its grid-stride loop suits long write streams.
  static constexpr bool vectorize_rows = min_bytes < 8;

  static bool wide(int64_t numel) {
    return prefer_wide ||
        numel / wide_vec >= waves * int64_t(at::xpu::getDeviceMaxWorkItems());
  }
};

template <int broadcast_arg, typename func_t, typename array_t>
static inline bool launch_broadcast_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    const array_t& data) {
  using config = BroadcastLaunchConfig<func_t, broadcast_arg>;
  const int ndim = iter.ndim();
  const int numel = iter.numel();
  if (ndim > 1 && !config::vectorize_rows) {
    return false;
  }
  const bool wide = config::wide(numel);
  const int vec_size = wide ? config::wide_vec : config::narrow_vec;
  if (ndim > 1 && iter.shape()[0] % vec_size != 0) {
    return false;
  }
  constexpr int dense = 2 - broadcast_arg;
  const int64_t vec_bytes = vec_size * iter.element_size(dense);
  if (reinterpret_cast<uintptr_t>(data[0]) %
              (vec_size * iter.element_size(0)) !=
          0 ||
      reinterpret_cast<uintptr_t>(data[dense]) % vec_bytes != 0) {
    return false;
  }
  for (int dim = 1; dim < ndim; ++dim) {
    if (iter.strides(dense)[dim] % vec_bytes != 0) {
      return false;
    }
  }

  auto submit = [&](auto vec,
                    auto bounds,
                    const auto& layout,
                    auto whole_waves) {
    constexpr int vec_v = decltype(vec)::value;
    constexpr bool full_waves = decltype(whole_waves)::value;
    using kernel_t = BroadcastElementwiseKernel<
        vec_v,
        broadcast_arg,
        config::prefer_wide,
        decltype(bounds)::value,
        func_t,
        std::decay_t<decltype(layout)>,
        full_waves>;
    constexpr int64_t wg_size = kernel_t::work_group_size;
    int64_t items = numel / vec_v + numel % vec_v;
    auto kernel_layout = layout;
    if constexpr (full_waves) {
      // A wide launch has at least one whole wave of vectors. A multiple of
      // the group size of work-items with numel - items vectors leaves
      // exactly one element per later work-item.
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
            reinterpret_cast<const typename kernel_t::dense_t*>(data[dense]),
            reinterpret_cast<const typename kernel_t::broadcast_t*>(
                data[broadcast_arg + 1]),
            reinterpret_cast<typename kernel_t::return_t*>(data[0]),
            f,
            kernel_layout));
  };
  auto launch = [&](auto vec, auto bounds, const auto& layout) {
    if constexpr (config::full_waves && decltype(vec)::value == 2) {
      // Whole vector waves keep the original kernel and its payload scheduling.
      if (numel % (2 * int64_t(at::xpu::getDeviceMaxWorkItems())) != 0) {
        submit(vec, bounds, layout, std::true_type{});
        return;
      }
    }
    submit(vec, bounds, layout, std::false_type{});
  };
  auto launch_at = [&](auto vec) {
    if constexpr (config::vectorize_rows) {
      if (ndim > 1) {
        launch(
            vec,
            std::true_type{},
            InnerBroadcast{
                uint32_t(numel), make_input_offset_calculator<2>(iter)});
        return;
      }
    }
    const SingleElementBroadcast<broadcast_arg> layout{uint32_t(numel)};
    if constexpr (config::prefer_wide) {
      if (numel %
              (decltype(vec)::value *
               launch_config::broadcast_wide_work_group_size) ==
          0) {
        launch(vec, std::false_type{}, layout);
        return;
      }
    }
    launch(vec, std::true_type{}, layout);
  };
  if constexpr (config::prefer_wide || config::wide_vec == config::narrow_vec) {
    launch_at(std::integral_constant<int, config::wide_vec>{});
  } else if (wide) {
    launch_at(std::integral_constant<int, config::wide_vec>{});
  } else {
    launch_at(std::integral_constant<int, config::narrow_vec>{});
  }
  return true;
}

// Takes binary iterators whose output is dense and one input of which
// broadcasts along the innermost dimension; see BroadcastElementwiseKernel.
template <typename func_t, typename array_t>
static inline bool try_launch_broadcast_kernel(
    TensorIteratorBase& iter,
    const func_t& f,
    const array_t& data) {
  static_assert(function_traits<func_t>::arity == 2);
  const int ndim = iter.ndim();
  const int numel = iter.numel();
  if (ndim == 0 || numel < 2) {
    return false;
  }
  const auto shape = iter.shape();
  if (!has_dense_output(iter, iter.element_size(0))) {
    return false;
  }
  int broadcast_arg;
  if (iter.strides(1)[0] == 0 && iter.strides(2)[0] == iter.element_size(2)) {
    broadcast_arg = 0;
  } else if (
      iter.strides(2)[0] == 0 && iter.strides(1)[0] == iter.element_size(1)) {
    broadcast_arg = 1;
  } else {
    return false;
  }
  // A broadcast input stored in the output would be read while written.
  const auto broadcast_begin =
      reinterpret_cast<uintptr_t>(data[broadcast_arg + 1]);
  auto broadcast_end = broadcast_begin + iter.element_size(broadcast_arg + 1);
  for (int dim = 0; dim < ndim; ++dim) {
    broadcast_end += (shape[dim] - 1) * iter.strides(broadcast_arg + 1)[dim];
  }
  if (overlaps_dense_output(iter, data[0], broadcast_begin, broadcast_end)) {
    return false;
  }
  return broadcast_arg == 0 ? launch_broadcast_kernel<0>(iter, f, data)
                            : launch_broadcast_kernel<1>(iter, f, data);
}

template <
    typename func_t,
    bool allow_legacy_strided = true,
    bool compact_unary = false>
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

  // Try contiguous indexing, inner broadcasts and aligned-row vectors,
  // then packed strided unary work before the general offset fallback.
  int vec_size;
  if (contiguous) {
    auto input_calc = TrivialOffsetCalculator<traits::arity>();
    vec_size = memory::can_vectorize_up_to<func_t>(data);
    launch_vectorized_kernel<compact_unary>(
        numel, f, data, input_calc, vec_size);
    return;
  } else {
    // The legacy row-vector path remains disabled for split iterators.
    if constexpr (traits::arity == 2) {
      if (try_launch_broadcast_kernel(iter, f, data)) {
        return;
      }
    }
    if constexpr (allow_legacy_strided) {
      if (!latency_case &&
          can_vectorize_for_non_contigouous<func_t>(iter, data, vec_size)) {
        auto input_calc = make_input_offset_calculator<traits::arity>(iter);
        launch_vectorized_kernel(numel, f, data, input_calc, vec_size);
        return;
      }
    }
    if constexpr (compact_unary) {
      if (try_launch_strided_input_unary_kernel(iter, f, data)) {
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
    typename func_t,
    bool allow_legacy_strided = true,
    bool compact_unary = false>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
  if (!needs_dynamic_casting<func_t>::check(iter)) {
    return gpu_kernel_impl_nocast<func_t, allow_legacy_strided, compact_unary>(
        iter, f);
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

// Unary functors, including tensor-scalar ops whose CPU scalar was captured
// into the functor, use the compact contiguous and strided-input launchers.
template <
    typename func_t,
    bool allow_legacy_strided = true,
    bool compact_unary = function_traits<func_t>::arity == 1>
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
      // Keep legacy row-vector restrictions for split iterators.
      // Iterator-local packed paths validate their own strides and pointers.
      gpu_kernel<func_t, false, compact_unary>(sub_iter, f);
    }
    return;
  }

  gpu_kernel_impl<func_t, allow_legacy_strided, compact_unary>(iter, f);
}

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
using AUnaryFunctor = ScalarUnaryFunctor<
    arg2_t,
    return_t,
    func_t,
    typename function_traits<func_t>::template arg<0>::type,
    true>;

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
using BUnaryFunctor = ScalarUnaryFunctor<
    arg1_t,
    return_t,
    func_t,
    typename function_traits<func_t>::template arg<1>::type,
    false>;

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BinaryFunctor {
  static constexpr bool prefer_wide_vectors =
      prefers_wide_vectors<func_t>::value;

  return_t operator()(arg1_t a, arg2_t b) const {
    return f(a, b);
  }
  BinaryFunctor(func_t f_) : f(f_) {}

 private:
  func_t f;
};

template <
    typename arg1_t,
    typename arg2_t = arg1_t,
    typename return_t = arg1_t,
    typename func_t,
    bool compact_unary = true>
void opmath_gpu_kernel_with_scalars(
    TensorIteratorBase& iter,
    const func_t& f,
    std::bool_constant<compact_unary> = {}) {
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
    gpu_kernel<decltype(af), true, compact_unary>(iter, af);
  } else if (iter.is_cpu_scalar(2)) {
    BUnaryFunctor<arg1_t, arg2_t, return_t, func_t> bf(
        f, iter.scalar_value<opmath_arg2_t>(2));
    iter.remove_operand(2);
    gpu_kernel<decltype(bf), true, compact_unary>(iter, bf);
  } else {
    gpu_kernel(iter, BinaryFunctor<arg1_t, arg2_t, return_t, func_t>(f));
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
    gpu_kernel(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
  } else {
    AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f, scalar_val);
    gpu_kernel(iter, unary_f);
  }
}

// Legacy variant that assumes that func_t has the correct types
// that we expect to load from memory
template <typename func_t, bool compact_unary = true>
void gpu_kernel_with_scalars(
    TensorIteratorBase& iter,
    const func_t& f,
    std::bool_constant<compact_unary> = {}) {
  using traits = function_traits<func_t>;
  static_assert(
      traits::arity == 2,
      "gpu_kernel_with_scalars only supports two input arguments");
  using arg1_t = typename traits::template arg<0>::type;
  using arg2_t = typename traits::template arg<1>::type;
  using return_t = typename traits::result_type;
  opmath_gpu_kernel_with_scalars<arg1_t, arg2_t, return_t, func_t>(
      iter, f, std::bool_constant<compact_unary>{});
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
