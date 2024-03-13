#pragma once

// #include <ATen/core/PhiloxRNGEngine.h>
#include <aten/sycl/MemoryAccess.h>
#include <aten/sycl/OffsetCalculator.h>
#include <aten/sycl/Philox4x32.h>
#include <comm/DeviceProperties.h>
#include <comm/Runtime.h>

namespace at {
namespace native {
namespace xpu {

using namespace at::xpu;

#define PHILOX_ENGINE_CALLS 4

struct PhiloxState {
  PhiloxState() = default;
  // Called if graph capture is not underway
  PhiloxState(uint64_t seed, uint64_t offset) {
    seed_ = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxState(
      uint64_t seed,
      int64_t* offset_extragraph,
      uint32_t offset_intragraph) {
    seed_ = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  uint64_t seed_ = 0;
  Payload offset_;
  uint32_t offset_intragraph_ = 0;
  bool captured_ = false;
};

inline std::tuple<uint64_t, uint64_t> philox_unpack(PhiloxState arg) {
  if (arg.captured_) {
    // static_cast avoids "warning: invalid narrowing conversion from "long" to
    // "unsigned long".
    // *(arg.offset_.ptr) is a broadcast load of a single int64_t to the entire
    // kernel. For most threads' reads it will hit in cache, so it shouldn't
    // hurt performance.
    return std::make_tuple(
        arg.seed_,
        static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
  } else {
    return std::make_tuple(arg.seed_, arg.offset_.val);
  }
}

inline std::tuple<uint64_t, uint32_t, uint32_t> calc_execution_policy(
    int64_t total_elements) {
  auto group_size = syclGpuHWThreadsPerEU() * syclMaxSubGroupSize();
  auto num_groups = (total_elements + group_size - 1) / group_size;
  auto hw_max_groups = syclMaxWorkItemsPerTile() / group_size;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;
  // number of times random will be generated per thread, to offset philox
  // counter in thc random state
  uint64_t counter_offset =
      ((total_elements - 1) / (group_size * num_groups * PHILOX_ENGINE_CALLS) +
       1) *
      PHILOX_ENGINE_CALLS;
  return std::make_tuple(counter_offset, num_groups, group_size);
}

// Just follow loops.h design
template <
    typename accscalar_t,
    int unroll_factor,
    typename dist_t,
    typename transform_t,
    typename item_t>
inline void distribution_elementwise_kernel(
    item_t& item,
    int numel,
    PhiloxState philox_args,
    dist_t dist_func,
    transform_t transform_func) {
  int group_size = item.get_local_range(0);
  int num_groups = item.get_group_range(0);
  int idx = item.get_group(0) * group_size + item.get_local_id(0);

  auto seeds = philox_unpack(philox_args);
  randStatePhilox4_32_10_t state;
  rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

  int full_tile_work_size = group_size * num_groups * unroll_factor;
  int rounded_size =
      ((numel - 1) / full_tile_work_size + 1) * full_tile_work_size;
  for (int linear_index = idx; linear_index < rounded_size;
       linear_index += full_tile_work_size) { // global range stride
    auto rand = dist_func(&state);
#pragma unroll
    for (int i = 0; i < unroll_factor; i++) {
      int li = linear_index + group_size * num_groups * i;
      if (li < numel) {
        transform_func(li, static_cast<accscalar_t>((&rand.x)[i]));
      }
    }
    // Some state (e.g. MTGP32) need to add barrier there.
  }
}

template <
    typename scalar_t,
    typename accscalar_t,
    int unroll_factor,
    typename dist_t,
    typename transform_t>
struct DistributionNullaryVecKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    distribution_elementwise_kernel<accscalar_t, unroll_factor>(
        item,
        numel_,
        PhiloxState(
            std::get<0>(rng_engine_inputs_), std::get<1>(rng_engine_inputs_)),
        dist_func_,
        [=](int idx, accscalar_t rand) {
          scalar_t* out = (scalar_t*)&out_data_[stride0_ * idx];
          *out = transform_func_(rand);
        });
  }
  DistributionNullaryVecKernelFunctor(
      int64_t numel,
      std::pair<uint64_t, uint64_t> rng_engine_inputs,
      dist_t dist_func,
      char* out_data,
      int stride0,
      transform_t transform_func)
      : numel_(numel),
        rng_engine_inputs_(rng_engine_inputs),
        dist_func_(dist_func),
        out_data_(out_data),
        stride0_(stride0),
        transform_func_(transform_func) {}

 private:
  int64_t numel_;
  std::pair<uint64_t, uint64_t> rng_engine_inputs_;
  dist_t dist_func_;
  char* out_data_;
  int stride0_;
  transform_t transform_func_;
};

template <
    typename scalar_t,
    typename accscalar_t,
    int unroll_factor,
    typename dist_t,
    typename transform_t>
struct DistributionNullaryUnrollKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    distribution_elementwise_kernel<accscalar_t, unroll_factor>(
        item,
        numel_,
        PhiloxState(
            std::get<0>(rng_engine_inputs_), std::get<1>(rng_engine_inputs_)),
        dist_func_,
        [=](int idx, accscalar_t rand) {
          auto offsets = offset_calc_.get(idx);
          scalar_t* out = (scalar_t*)&out_data_[offsets[0]];
          *out = transform_func_(rand);
        });
  }
  DistributionNullaryUnrollKernelFunctor(
      int64_t numel,
      std::pair<uint64_t, uint64_t> rng_engine_inputs,
      dist_t dist_func,
      char* out_data,
      transform_t transform_func,
      OffsetCalculator<1> offset_calc)
      : numel_(numel),
        rng_engine_inputs_(rng_engine_inputs),
        dist_func_(dist_func),
        out_data_(out_data),
        transform_func_(transform_func),
        offset_calc_(offset_calc) {}

 private:
  int64_t numel_;
  std::pair<uint64_t, uint64_t> rng_engine_inputs_;
  dist_t dist_func_;
  char* out_data_;
  transform_t transform_func_;
  OffsetCalculator<1> offset_calc_;
};

template <
    typename scalar_t,
    typename accscalar_t,
    int unroll_factor,
    typename RNG,
    typename dist_t,
    typename transform_t>
void distribution_nullary_kernel(
    at::TensorIteratorBase& iter,
    RNG gen,
    const dist_t dist_func,
    transform_t transform_func) {
  static_assert(unroll_factor >= 1, "unroll_factor must be >= 1.");
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  auto execution_policy = calc_execution_policy(numel);
  auto counter_offset = std::get<0>(execution_policy);
  auto num_groups = std::get<1>(execution_policy);
  auto group_size = std::get<2>(execution_policy);

  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(counter_offset);
  }

  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<scalar_t, accscalar_t, unroll_factor>(
          sub_iter, gen, dist_func, transform_func);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);

  if (iter.is_trivial_1d()) {
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    auto caller = DistributionNullaryVecKernelFunctor<
        scalar_t,
        accscalar_t,
        unroll_factor,
        dist_t,
        transform_t>(
        numel, rng_engine_inputs, dist_func, out_data, stride0, transform_func);
    sycl_kernel_submit(
        num_groups * group_size, group_size, getCurrentSYCLQueue(), caller);
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    auto caller = DistributionNullaryUnrollKernelFunctor<
        scalar_t,
        accscalar_t,
        unroll_factor,
        dist_t,
        transform_t>(
        numel,
        rng_engine_inputs,
        dist_func,
        out_data,
        transform_func,
        offset_calc);
    sycl_kernel_submit(
        num_groups * group_size, group_size, getCurrentSYCLQueue(), caller);
  }
}

struct Uniform2DistributionFunctor {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    return rand_uniform2_double(state);
  }
};

struct Uniform4DistributionFunctor {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    return rand_uniform4(state);
  }
};

template <
    typename scalar_t,
    typename accscalar_t,
    size_t engine_calls,
    typename RNG,
    typename transform_t>
void uniform_and_transform(
    TensorIteratorBase& iter,
    RNG gen,
    transform_t transform) {
  // Distribution backbone only handle two accumulate type.
  if (std::is_same<scalar_t, double>::value) {
    Uniform2DistributionFunctor f;
    distribution_nullary_kernel<scalar_t, accscalar_t, engine_calls / 2>(
        iter, gen, f, transform);
  } else {
    Uniform4DistributionFunctor f;
    distribution_nullary_kernel<scalar_t, accscalar_t, engine_calls>(
        iter, gen, f, transform);
  }
}

struct Normal2DistributionFunctor {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    return rand_normal2_double(state);
  }
};

struct Normal4DistributionFunctor {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    return rand_normal4(state);
  }
};

template <
    typename scalar_t,
    typename accscalar_t,
    size_t engine_calls,
    typename RNG,
    typename transform_t>
void normal_and_transform(
    TensorIteratorBase& iter,
    RNG gen,
    transform_t transform) {
  if (std::is_same<scalar_t, double>::value) {
    Normal2DistributionFunctor f;
    distribution_nullary_kernel<scalar_t, accscalar_t, engine_calls / 2>(
        iter, gen, f, transform);
  } else {
    Normal4DistributionFunctor f;
    distribution_nullary_kernel<scalar_t, accscalar_t, engine_calls>(
        iter, gen, f, transform);
  }
}

} // namespace xpu
} // namespace native
} // namespace at
