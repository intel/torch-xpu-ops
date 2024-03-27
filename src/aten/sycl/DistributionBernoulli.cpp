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

template <typename scalar_t>
struct BernoulliCompareFunctor {
  scalar_t operator()(scalar_t out, scalar_t p) const {
    return static_cast<scalar_t>(out < p);
  }
};

template <typename scalar_t, typename accscalar_t>
struct BernoulliCompareScalarFunctor {
  scalar_t operator()(accscalar_t rand) const {
    return static_cast<scalar_t>(rand < static_cast<accscalar_t>(p_));
  }

  BernoulliCompareScalarFunctor(accscalar_t p) : p_(p) {}

 private:
  accscalar_t p_;
};

void bernoulli_compare_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.dtype(),
      "bernoulli_xpu",
      [&] {
        auto f = BernoulliCompareFunctor<scalar_t>();
        gpu_kernel(iter, f);
      });
}

void bernoulli_compare_scalar_kernel(
    TensorIterator& iter,
    double p,
    c10::optional<Generator> gen) {
  auto gen_ = get_generator_or_default<at::XPUGeneratorImpl>(
      gen, at::xpu::detail::getDefaultXPUGenerator());
  AT_DISPATCH_ALL_TYPES_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.dtype(),
      "bernoulli_scalar_xpu",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        auto p_ = static_cast<accscalar_t>(p);
        auto f = BernoulliCompareScalarFunctor<scalar_t, accscalar_t>(p_);
        uniform_and_transform<scalar_t, accscalar_t, PHILOX_ENGINE_CALLS>(
            iter, gen_, f);
      });
}

} // namespace xpu
} // namespace native
} // namespace at
