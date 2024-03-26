#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/DistributionKernels.h>

namespace at {

Tensor& XPUNativeFunctions::normal_(
    Tensor& self,
    double mean,
    double std,
    const ::std::optional<at::Generator>& generator) {
  TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std);
  if (self.is_complex()) {
    auto float_tensor = at::view_as_real(self);
    // variance for normal distribution of the real and imaginary values
    // is half of the input variance
    auto iter = TensorIterator::nullary_op(float_tensor);
    native::xpu::normal_kernel(iter, mean, std / (std::sqrt(2)), generator);
  } else {
    auto iter = TensorIterator::nullary_op(self);
    native::xpu::normal_kernel(iter, mean, std, generator);
  }
  return self;
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
  TORCH_CHECK(var >= min && var <= max, name, " is out of bounds for ", dtype);

Tensor& XPUNativeFunctions::uniform_(
    Tensor& self,
    double from,
    double to,
    const ::std::optional<at::Generator>& generator) {
  if (self.is_complex()) {
    auto float_tensor = at::view_as_real(self);
    uniform_(float_tensor, from, to, generator);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "check_uniform_bounds",
        [&] {
          const auto dtype = self.dtype();
          const auto min =
              static_cast<double>(std::numeric_limits<scalar_t>::lowest());
          const auto max =
              static_cast<double>(std::numeric_limits<scalar_t>::max());
          CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
          CHECK_OUT_OF_BOUNDS(to, "to", min, max, dtype);
          TORCH_CHECK(
              from <= to,
              "uniform_ expects to return a [from, to) range, but found from=",
              from,
              " > to=",
              to);
          TORCH_CHECK(
              (to - from) <= std::numeric_limits<scalar_t>::max(),
              "uniform_ expects to-from <= std::numeric_limits<",
              toString(self.scalar_type()),
              ">::max(), but found to=",
              to,
              " and from=",
              from,
              " which result in to-from to exceed the limit");
          from = std::min(std::max(from, min), max);
          to = std::max(std::min(to, max), min);
        });
    auto iter = at::TensorIterator::nullary_op(self);
    native::xpu::uniform_kernel(iter, from, to, generator);
  }
  return self;
}
} // namespace at
