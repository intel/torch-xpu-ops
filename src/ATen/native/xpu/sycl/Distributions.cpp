#include <ATen/Dispatch.h>
#include <ATen/native/Distributions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <ATen/native/xpu/sycl/Distributions.h>
#include <ATen/native/xpu/sycl/TensorApplyUtils.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

namespace at::native::xpu {

template <typename scalar_t>
struct PoissonTensorApplyFunctor {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t& ret_val,
      const scalar_t& lambda) const {
    SYCL_KERNEL_ASSERT(
        lambda >= 0 &&
        "invalid Poisson rate, expected rate to be non-negative");
    auto seeds = philox_unpack(philox_args_);
    randStatePhilox4_32_10_t state;
    rand_init(
        std::get<0>(seeds),
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0),
        std::get<1>(seeds),
        &state);
    ret_val = static_cast<scalar_t>(rand_poisson(&state, lambda));
  }
  PoissonTensorApplyFunctor(std::pair<uint64_t, uint64_t> rng_engine_inputs)
      : philox_args_(
            std::get<0>(rng_engine_inputs),
            std::get<1>(rng_engine_inputs)) {}

 private:
  PhiloxState philox_args_;
};

template <typename scalar_t>
void poisson_kernel(
    const at::TensorBase& ret,
    const at::TensorBase& lambda,
    std::pair<uint64_t, uint64_t> rng_engine_inputs) {
  auto functor = PoissonTensorApplyFunctor<scalar_t>(rng_engine_inputs);
  at::native::xpu::tensor_apply2<
      scalar_t,
      scalar_t,
      decltype(functor),
      /*max_threads_per_block=*/512>(
      const_cast<at::TensorBase&>(ret),
      const_cast<at::TensorBase&>(lambda),
      functor);
}

void launch_poisson_kernel(
    const TensorBase& ret,
    const TensorBase& lambda,
    at::XPUGeneratorImpl* gen) {
  std::pair<uint64_t, uint64_t> rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_engine_inputs(20);
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      ret.scalar_type(),
      "poisson_xpu",
      [&] { poisson_kernel<scalar_t>(ret, lambda, rng_engine_inputs); });
}

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
void binomial_kernel(TensorIteratorBase& iter, PhiloxState philox_args) {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  BinomialFunctor<scalar_t, accscalar_t> f;
  at::native::xpu::distribution_binary_kernel(iter, philox_args, f);
}

void launch_binomial_kernel(TensorIteratorBase& iter, XPUGeneratorImpl* gen) {
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
      [&] { binomial_kernel<scalar_t>(iter, rng_engine_inputs); });
}

template <typename scalar_t, typename accscalar_t>
struct GammaTensorApplyFunctor {
  void operator()(
      sycl::nd_item<1> item,
      scalar_t& ret_val,
      const scalar_t& alpha) const {
    auto seeds = philox_unpack(philox_args_);
    randStatePhilox4_32_10_t state;
    rand_init(
        std::get<0>(seeds),
        item.get_group(0) * item.get_local_range(0) + item.get_local_id(0),
        std::get<1>(seeds),
        &state);

    auto uniform_lambda = [&state]() { return rand_uniform(&state); };
    BaseSampler<accscalar_t, decltype(uniform_lambda)> standard_uniform(
        uniform_lambda);

    auto normal_lambda = [&state]() { return rand_normal(&state); };
    BaseSampler<accscalar_t, decltype(normal_lambda)> standard_normal(
        normal_lambda);

    auto sample = sample_gamma<
        scalar_t,
        accscalar_t,
        decltype(uniform_lambda),
        decltype(normal_lambda)>(alpha, standard_uniform, standard_normal);
    auto min_value = std::numeric_limits<scalar_t>::min();
    ret_val = (min_value > sample) ? min_value : sample;
  }

  GammaTensorApplyFunctor(PhiloxState philox_args)
      : philox_args_(philox_args) {}

 private:
  PhiloxState philox_args_;
};

template <typename scalar_t>
void gamma_kernel(
    const at::TensorBase& ret,
    const at::TensorBase& alpha,
    PhiloxState philox_args) {
  using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
  GammaTensorApplyFunctor<scalar_t, accscalar_t> functor(philox_args);
  at::native::xpu::tensor_apply2<
      scalar_t,
      scalar_t,
      decltype(functor),
      /*max_threads_per_block=*/512>(
      const_cast<at::TensorBase&>(ret),
      const_cast<at::TensorBase&>(alpha),
      functor);
}

void launch_gamma_kernel(
    Tensor& ret,
    const Tensor& alpha,
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
      ret.scalar_type(),
      "gamma_xpu",
      [&] { gamma_kernel<scalar_t>(ret, alpha, rng_engine_inputs); });
}

template <typename scalar_t, typename accscalar_t>
struct StandardGammaGradKernelFunctor {
  scalar_t operator()(scalar_t self_val, scalar_t output_val) const {
    return standard_gamma_grad_one<scalar_t, accscalar_t>(self_val, output_val);
  }
};

void launch_standard_gamma_grad_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.input_dtype(),
      "_standard_gamma_grad_xpu",
      [&] {
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
        StandardGammaGradKernelFunctor<scalar_t, accscalar_t> f;
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t>
struct DirichletKernelFunctor {
  scalar_t operator()(scalar_t gamma, scalar_t gamma_sum) const {
    auto ret_val = gamma / gamma_sum;
    auto min_value = std::numeric_limits<scalar_t>::min();
    auto max_value = 1 - std::numeric_limits<scalar_t>::epsilon();
    ret_val = (min_value > ret_val) ? min_value : ret_val;
    ret_val = (max_value < ret_val) ? max_value : ret_val;
    return ret_val;
  }
};

void launch_dirichlet_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.input_dtype(),
      "dirichlet_xpu",
      [&] {
        DirichletKernelFunctor<scalar_t> f;
        gpu_kernel(iter, f);
      });
}

template <typename scalar_t, typename accscalar_t>
struct DirichletGradKernelFunctor {
  scalar_t operator()(scalar_t x_val, scalar_t alpha_val, scalar_t total_val)
      const {
    return dirichlet_grad_one<scalar_t, accscalar_t>(
        x_val, alpha_val, total_val);
  }
};

void launch_dirichlet_grad_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.input_dtype(), "_dirichlet_grad_xpu", [&] {
    using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
    DirichletGradKernelFunctor<scalar_t, accscalar_t> f;
    gpu_kernel(iter, f);
  });
}

} // namespace at::native::xpu
