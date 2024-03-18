#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUGeneratorImpl.h>
#include <aten/sycl/DistributionKernelTemplates.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/MemoryAccess.h>
#include <aten/sycl/OffsetCalculator.h>
#include <aten/sycl/Philox4x32.h>
#include <comm/DeviceProperties.h>
#include <comm/Runtime.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t, typename accscalar_t>
struct UniformTransformFunctor {
  scalar_t operator()(accscalar_t rand) const {
    auto reverse_bound_rand = rand == static_cast<accscalar_t>(1.0)
        ? static_cast<accscalar_t>(0.0)
        : rand;
    return static_cast<scalar_t>(reverse_bound_rand * range_ + from_);
  }
  UniformTransformFunctor(accscalar_t range, scalar_t from)
      : range_(range), from_(from) {}

 private:
  accscalar_t range_;
  scalar_t from_;
};

void uniform_kernel(
    TensorIterator& iter,
    double from,
    double to,
    c10::optional<Generator> gen) {
  auto gen_ = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "uniform_xpu",
      [&] {
        auto from_ = static_cast<scalar_t>(from);
        auto to_ = static_cast<scalar_t>(to);
        using accscalar_t = acc_type<scalar_t, false>;
        auto range = static_cast<accscalar_t>(to_ - from_);
        auto f = UniformTransformFunctor<scalar_t, accscalar_t>(range, from_);
        uniform_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen_, f);
      });
}

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

void normal_kernel(
    TensorIterator& iter,
    double mean,
    double std,
    c10::optional<Generator> gen) {
  auto gen_ = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "normal_xpu",
      [&] {
        using accscalar_t = acc_type<scalar_t, false>;
        auto mean_ = static_cast<accscalar_t>(mean);
        auto std_ = static_cast<accscalar_t>(std);
        // define lambda to multiply std and add mean
        auto f = NormalTransformFunctor<scalar_t, accscalar_t>(mean_, std_);
        normal_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen_, f);
      });
}

template <typename scalar_t>
struct BernoulliCompareFunctor {
  scalar_t operator()(scalar_t out, scalar_t p) const {
    return static_cast<scalar_t>(out < p);
  }
};

void bernoulli_compare_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "bernoulli_xpu",
      [&] {
        auto f = BernoulliCompareFunctor<scalar_t>();
        gpu_kernel(iter, f);
      });
}

} // namespace xpu
} // namespace native
} // namespace at
