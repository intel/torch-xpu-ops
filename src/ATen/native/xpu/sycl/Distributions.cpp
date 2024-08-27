#include <ATen/Dispatch.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

struct rand_uniform_wrapper {
  rand_uniform_wrapper(randStatePhilox4_32_10_t& state) : state_(state) {}

  float operator()() {
    uint32_t val = rand(&state_); // need just bits
    constexpr auto MASK = static_cast<uint32_t>(
        (static_cast<uint64_t>(1) << std::numeric_limits<float>::digits) - 1);
    constexpr auto DIVISOR = static_cast<float>(1) /
        (static_cast<uint32_t>(1) << std::numeric_limits<float>::digits);
    return (val & MASK) * DIVISOR;
  }

  randStatePhilox4_32_10_t& state_;
};

template <typename scalar_t, typename accscalar_t>
struct BinomialFunctor {
  scalar_t operator()(
      randStatePhilox4_32_10_t& state,
      scalar_t count,
      scalar_t prob) const {
    auto uniform_lambda = rand_uniform_wrapper(state);
    BaseSampler<accscalar_t, decltype(uniform_lambda)> standard_uniform(
        uniform_lambda);
    auto sample =
        sample_binomial<scalar_t, accscalar_t, decltype(uniform_lambda)>(
            count, prob, standard_uniform);
    return static_cast<scalar_t>(sample);
  }
};

template <typename scalar_t>
void binomial_xpu_kernel(TensorIteratorBase& iter, PhiloxState philox_args) {
  using accscalar_t = at::acc_type<scalar_t, true>;
  BinomialFunctor<scalar_t, accscalar_t> f;
  at::native::xpu::distribution_binary_kernel(iter, philox_args, f);
}

void launch_binomial_xpu_kernel(
    TensorIteratorBase& iter,
    XPUGeneratorImpl* gen) {
  std::pair<uint64_t, uint64_t> engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    engine_inputs = gen->philox_engine_inputs(42);
  }
  PhiloxState rng_engine_inputs(
      std::get<0>(engine_inputs), std::get<1>(engine_inputs));
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.input_dtype(),
      "binomial_xpu",
      [&] { binomial_xpu_kernel<scalar_t>(iter, rng_engine_inputs); });
}

} // namespace at::native::xpu
