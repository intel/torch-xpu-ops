#pragma once

#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ExpandBase.h>
#include <ATen/OpMathType.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/MemoryAccess.h>
#include <aten/sycl/OffsetCalculator.h>
#include <aten/sycl/Philox4x32.h>
#include <comm/DeviceProperties.h>
#include <comm/Runtime.h>

namespace at {
namespace native {
namespace xpu {

using namespace at::xpu;

const uint32_t rand4_engine_calls = 4;

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

template <uint32_t UNROLL = rand4_engine_calls>
inline std::tuple<uint64_t, uint32_t, uint32_t> calc_execution_policy(
    int64_t total_elements) {
  auto group_size =
      syclMaxWorkItemsPerEU(); // TODO: see
                               // https://github.com/intel/torch-xpu-ops/issues/135
  auto num_groups = (total_elements + group_size - 1) / group_size;
  auto hw_max_groups = syclMaxWorkItemsPerTile() / group_size;
  num_groups = num_groups > hw_max_groups ? hw_max_groups : num_groups;
  // number of times random will be generated per thread, to offset philox
  // counter in thc random state
  uint64_t counter_offset =
      ((total_elements - 1) / (group_size * num_groups * UNROLL) + 1) * UNROLL;
  return std::make_tuple(counter_offset, num_groups, group_size);
}

template <
    typename scalar_t,
    typename accscalar_t,
    int unroll_factor,
    typename dist_t,
    typename transform_t,
    typename offset_calc_t>
struct DistributionElementwiseKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    int group_size = item.get_local_range(0);
    int num_groups = item.get_group_range(0);
    int idx = item.get_global_linear_id();

    auto seeds = philox_unpack(philox_args_);
    randStatePhilox4_32_10_t state;
    rand_init(std::get<0>(seeds), idx, std::get<1>(seeds), &state);

    int full_tile_work_size = group_size * num_groups * unroll_factor;
    int rounded_size =
        ((numel_ - 1) / full_tile_work_size + 1) * full_tile_work_size;
    for (int linear_index = idx; linear_index < rounded_size;
         linear_index += full_tile_work_size) { // global range stride
      auto rand = dist_func_(&state);
#pragma unroll
      for (int i = 0; i < unroll_factor; i++) {
        int li = linear_index + group_size * num_groups * i;
        if (li < numel_) {
          if constexpr (std::is_integral<offset_calc_t>::value) {
            scalar_t* out = (scalar_t*)&out_data_[offset_calc_ * li];
            *out = transform_func_(static_cast<accscalar_t>((&rand.x)[i]));
          } else {
            auto offsets = offset_calc_.get(li);
            scalar_t* out = (scalar_t*)&out_data_[offsets[0]];
            *out = transform_func_(static_cast<accscalar_t>((&rand.x)[i]));
          }
        }
      }
      // Some state (e.g. MTGP32) need to add barrier there.
    }
  }
  DistributionElementwiseKernelFunctor(
      int64_t numel,
      std::pair<uint64_t, uint64_t> rng_engine_inputs,
      dist_t dist_func,
      transform_t transform_func,
      char* out_data,
      offset_calc_t offset_calc)
      : numel_(numel),
        philox_args_(PhiloxState(
            std::get<0>(rng_engine_inputs),
            std::get<1>(rng_engine_inputs))),
        dist_func_(dist_func),
        transform_func_(transform_func),
        out_data_(out_data),
        offset_calc_(offset_calc) {}

 private:
  int64_t numel_;
  PhiloxState philox_args_;
  dist_t dist_func_;
  transform_t transform_func_;
  char* out_data_;
  offset_calc_t offset_calc_;
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
    auto caller = DistributionElementwiseKernelFunctor<
        scalar_t,
        accscalar_t,
        unroll_factor,
        dist_t,
        transform_t,
        decltype(stride0)>(
        numel, rng_engine_inputs, dist_func, transform_func, out_data, stride0);
    sycl_kernel_submit(
        num_groups * group_size, group_size, getCurrentSYCLQueue(), caller);
  } else {
    auto offset_calc = make_offset_calculator<1>(iter);
    auto caller = DistributionElementwiseKernelFunctor<
        scalar_t,
        accscalar_t,
        unroll_factor,
        dist_t,
        transform_t,
        decltype(offset_calc)>(
        numel,
        rng_engine_inputs,
        dist_func,
        transform_func,
        out_data,
        offset_calc);
    sycl_kernel_submit(
        num_groups * group_size, group_size, getCurrentSYCLQueue(), caller);
  }
}

} // namespace xpu
} // namespace native
} // namespace at

namespace at {
namespace native {
namespace templates {
namespace xpu {

using namespace at::native::xpu;

// ====================== Random ======================

struct Rand4ULL2DistFunctor {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    ulonglong2 ret;
    uint4 rand_val = rand4(state);
    ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
    ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
    return ret;
  }
};

struct Rand4DistFunctor {
  auto operator()(randStatePhilox4_32_10_t* state) const {
    return rand4(state);
  }
};

template <typename T, typename V>
struct UniformIntFromToTransformFunctor {
  auto operator()(V val) const {
    return transformation::uniform_int_from_to<T, V>(val, range_, base_);
  }
  UniformIntFromToTransformFunctor(uint64_t range, int64_t base)
      : range_(range), base_(base) {}

 private:
  uint64_t range_;
  int64_t base_;
};

template <typename RNG>
void random_from_to_kernel(
    TensorIteratorBase& iter,
    uint64_t range,
    int64_t base,
    RNG gen) {
  AT_DISPATCH_V2(
      iter.dtype(),
      "random_from_to_kernel_xpu",
      AT_WRAP([&] {
        if ((std::is_same<scalar_t, int64_t>::value ||
             std::is_same<scalar_t, double>::value ||
             std::is_same<scalar_t, float>::value ||
             std::is_same<scalar_t, at::BFloat16>::value) &&
            range >= 1ULL << 32) {
          distribution_nullary_kernel<
              scalar_t,
              uint64_t,
              rand4_engine_calls / 2>(
              iter,
              gen,
              Rand4ULL2DistFunctor(),
              UniformIntFromToTransformFunctor<scalar_t, uint64_t>(
                  range, base));
        } else {
          distribution_nullary_kernel<scalar_t, uint32_t, rand4_engine_calls>(
              iter,
              gen,
              Rand4DistFunctor(),
              UniformIntFromToTransformFunctor<scalar_t, uint32_t>(
                  range, base));
        }
      }),
      AT_EXPAND(AT_ALL_TYPES),
      kBool,
      kHalf,
      kBFloat16,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

template <typename T, typename V>
struct UniformIntFullRangeTransformFunctor {
  auto operator()(V val) const {
    return at::transformation::uniform_int_full_range<T, V>(val);
  }
};

template <typename RNG>
void random_full_64_bits_range_kernel(TensorIteratorBase& iter, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16,
      iter.dtype(),
      "random_full_64_bits_range_kernel_xpu",
      [&] {
        if (std::is_same<scalar_t, int64_t>::value ||
            std::is_same<scalar_t, double>::value ||
            std::is_same<scalar_t, float>::value ||
            std::is_same<scalar_t, at::BFloat16>::value) {
          distribution_nullary_kernel<
              scalar_t,
              uint64_t,
              rand4_engine_calls / 2>(
              iter,
              gen,
              Rand4ULL2DistFunctor(),
              UniformIntFullRangeTransformFunctor<scalar_t, uint64_t>());
        } else {
          TORCH_CHECK(
              false,
              "random_full_64_bits_range_kernel_xpu handles only int64, double, float and bfloat16");
        }
      });
}

template <typename T, typename V>
struct UniformIntTransformFunctor {
  auto operator()(V val) const {
    return transformation::uniform_int<T, V>(val);
  }
};

template <typename RNG>
void random_kernel(TensorIteratorBase& iter, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.dtype(),
      "random_kernel_xpu",
      [&] {
        if (std::is_same<scalar_t, double>::value ||
            std::is_same<scalar_t, int64_t>::value) {
          distribution_nullary_kernel<
              scalar_t,
              uint64_t,
              rand4_engine_calls / 2>(
              iter,
              gen,
              Rand4ULL2DistFunctor(),
              UniformIntTransformFunctor<scalar_t, uint64_t>());
        } else {
          distribution_nullary_kernel<scalar_t, uint32_t, rand4_engine_calls>(
              iter,
              gen,
              Rand4DistFunctor(),
              UniformIntTransformFunctor<scalar_t, uint32_t>());
        }
      });
}

// ============================================

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

// ====================== Normal ======================

template <typename scalar_t, typename accscalar_t>
struct NormalTransformFunctor {
  scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(rand * std_ + mean_);
  }
  NormalTransformFunctor(accscalar_t mean, accscalar_t std)
      : mean_(mean), std_(std) {}

 private:
  accscalar_t mean_;
  accscalar_t std_;
};

template <typename RNG>
void normal_kernel(const TensorBase& self, double mean_, double std_, RNG gen) {
  auto iter = TensorIterator::borrowing_nullary_op(self);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "normal_kernel_xpu",
      [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        auto mean = static_cast<accscalar_t>(mean_);
        auto std = static_cast<accscalar_t>(std_);
        normal_and_transform<scalar_t, accscalar_t, rand4_engine_calls>(
            iter,
            gen,
            NormalTransformFunctor<scalar_t, accscalar_t>(mean, std));
      });
}

// ====================== Uniform ======================

template <typename scalar_t, typename opmath_t>
struct UniformTransformFunctor {
  scalar_t operator()(opmath_t rand) const {
    auto value = static_cast<scalar_t>(rand * range_ + from_);
    auto reverse_bound_value = value == to_ ? from_ : value;
    return reverse_bound_value;
  }
  UniformTransformFunctor(opmath_t range, scalar_t from, scalar_t to)
      : range_(range), from_(from), to_(to) {}

 private:
  opmath_t range_;
  scalar_t from_;
  scalar_t to_;
};

template <typename RNG>
void uniform_kernel(
    TensorIteratorBase& iter,
    double from_,
    double to_,
    RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "uniform_kernel_xpu",
      [&] {
        auto from = static_cast<scalar_t>(from_);
        auto to = static_cast<scalar_t>(to_);
        using opmath_t = at::opmath_type<scalar_t>;
        auto range = static_cast<opmath_t>(to - from);
        uniform_and_transform<scalar_t, opmath_t, rand4_engine_calls>(
            iter,
            gen,
            UniformTransformFunctor<scalar_t, opmath_t>(range, from, to));
      });
}

// ====================== Bernoulli ======================

template <typename scalar_t, typename accscalar_t>
struct BernoulliFunctor {
  scalar_t operator()(scalar_t out, accscalar_t p) const {
    return static_cast<scalar_t>((accscalar_t)out < p);
  }
};

template <typename RNG>
void bernoulli_kernel(const TensorBase& self, const TensorBase& p_, RNG gen) {
  TORCH_CHECK(
      at::isFloatingType(p_.scalar_type()),
      "expected probabilities tensor to have floating type, got ",
      p_.scalar_type());
  const auto p_type = self.dtype() == at::kDouble ? at::kDouble : at::kFloat;
  auto p_xpu = p_.to(TensorOptions().device(self.device()).dtype(p_type));
  auto p = expand_inplace(self, p_xpu);

  Tensor self_float;
  auto self_type = self.scalar_type();
  if (!(self_type == at::ScalarType::Float ||
        self_type == at::ScalarType::Double))
    self_float = at::empty(self.sizes(), self.options().dtype(at::kFloat));
  else
    self_float = self;

  auto iter_uniform = at::TensorIterator::borrowing_nullary_op(self_float);
  uniform_kernel<RNG>(iter_uniform, 0.0, 1.0, gen);

  auto iter = TensorIteratorConfig()
                  .add_output(self)
                  .add_input(self_float)
                  .add_input(*p)
                  .check_all_same_dtype(false)
                  .build();

  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self.scalar_type(),
      "bernoulli_xpu",
      [&] {
        using accscalar_t = at::DiscreteDistributionType<scalar_t>::type;
        auto f = BernoulliFunctor<scalar_t, accscalar_t>();
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t, typename accscalar_t>
struct BernoulliScalarFunctor {
  scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(
        transformation::bernoulli<accscalar_t>(rand, p_));
  }
  BernoulliScalarFunctor(double p) : p_(p) {}

 private:
  double p_;
};

template <typename RNG>
void bernoulli_kernel(TensorIteratorBase& iter, double p, RNG gen) {
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.dtype(),
      "bernoulli_scalar_xpu_",
      [&] {
        using accscalar_t = at::DiscreteDistributionType<scalar_t>::type;
        uniform_and_transform<scalar_t, accscalar_t, rand4_engine_calls>(
            iter, gen, BernoulliScalarFunctor<scalar_t, accscalar_t>(p));
      });
}

} // namespace xpu
} // namespace templates
} // namespace native
} // namespace at
