#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/ScalarOps.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DistributionTemplates.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/DistributionKernels.h>

namespace at {

Tensor& XPUNativeFunctions::normal_(
    Tensor& self,
    double mean,
    double std,
    ::std::optional<at::Generator> generator) {
  TORCH_CHECK(std >= 0.0, "normal_ expects std >= 0.0, but found std=", std);
  if (self.is_complex()) {
    auto float_tensor = at::view_as_real(self);
    // variance for normal distribution of the real and imaginary values
    // is half of the input variance
    native::xpu::normal_kernel(
        float_tensor, mean, std / (std::sqrt(2)), generator);
  } else {
    native::xpu::normal_kernel(self, mean, std, generator);
  }
  return self;
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
  TORCH_CHECK(var >= min && var <= max, name, " is out of bounds for ", dtype);

#define CHECK_EMPTY_AND_RETURN(tensor) \
  if (tensor.numel() == 0) {           \
    return tensor;                     \
  }

Tensor& XPUNativeFunctions::uniform_(
    Tensor& self,
    double from,
    double to,
    ::std::optional<at::Generator> generator) {
  if (self.is_complex()) {
    auto float_tensor = at::view_as_real(self);
    uniform_(float_tensor, from, to, generator);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self.scalar_type(),
        "check_uniform_bounds_xpu",
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
    CHECK_EMPTY_AND_RETURN(self);
    auto iter = at::TensorIterator::borrowing_nullary_op(self);
    native::xpu::uniform_kernel(iter, from, to, generator);
  }
  return self;
}

Tensor& XPUNativeFunctions::bernoulli_(
    Tensor& self,
    const Tensor& p_,
    ::std::optional<Generator> generator) {
  native::xpu::bernoulli_tensor_kernel(self, p_, std::move(generator));
  return self;
}

Tensor& XPUNativeFunctions::bernoulli_(
    Tensor& self,
    double p,
    ::std::optional<Generator> generator) {
  native::xpu::bernoulli_scalar_kernel(self, p, std::move(generator));
  return self;
}

#define CHECK_OUT_OF_BOUNDS(var, name, min, max, dtype) \
  TORCH_CHECK(var >= min && var <= max, name, " is out of bounds for ", dtype);

#define WARN_OUT_OF_BOUNDS(var, name, digits, dtype)                                                 \
  if (var < -(1LL << digits) || var > (1LL << digits)) {                                             \
    TORCH_WARN(                                                                                      \
        name,                                                                                        \
        " is out of bounds [-(2^",                                                                   \
        digits,                                                                                      \
        "), 2^",                                                                                     \
        digits,                                                                                      \
        "]. ",                                                                                       \
        "Due to precision limitations ",                                                             \
        dtype,                                                                                       \
        " can support discrete uniform distribution only within this range. ",                       \
        "This warning will become an error in version 1.7 release, please fix the code in advance"); \
  }

Tensor& XPUNativeFunctions::random_(
    Tensor& self,
    int64_t from,
    c10::optional<int64_t> to_opt,
    ::std::optional<Generator> generator) {
  uint64_t range = 0;
  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  if (to_opt.has_value()) {
    // [from, to)
    int64_t to = *to_opt;
    TORCH_CHECK(
        from < to,
        "random_ expects 'from' to be less than 'to', but got from=",
        from,
        " >= to=",
        to);
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          self.scalar_type(),
          "random_update_from_to",
          [&] {
            from = native::templates::update_from<scalar_t>(from);
            to = native::templates::update_to<scalar_t>(to);
            TORCH_CHECK(
                from < to,
                "random_ expects 'from' casted to dtype to be less than 'to' casted to dtype, but got from=",
                from,
                " >= to=",
                to);
          });
    }
    native::templates::check_from_to_in_range(from, to - 1, self.dtype());
    CHECK_EMPTY_AND_RETURN(self);
    range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);
    native::xpu::random_from_to_kernel(iter, range, from, generator);
  } else if (from != std::numeric_limits<int64_t>::lowest()) {
    // [from, std::numeric_limits<int64_t>::max()]
    int64_t to_inc = 0;
    if (isFloatingType(iter.dtype())) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          self.scalar_type(),
          "random_from_to_range_calc",
          [&] {
            constexpr int64_t scalar_t_max = static_cast<int64_t>(1)
                << std::numeric_limits<scalar_t>::digits;
            to_inc = scalar_t_max > std::numeric_limits<int64_t>::max()
                ? std::numeric_limits<int64_t>::max()
                : static_cast<int64_t>(scalar_t_max);
            from = native::templates::update_from<scalar_t>(from);
            TORCH_CHECK(
                from < to_inc,
                "random_ expects 'from' casted to dtype to be less than or equal to 'to_inc' casted to dtype, but got from=",
                from,
                " > to_inc=",
                to_inc);
          });
    } else if (isIntegralType(iter.dtype(), /*includeBool=*/true)) {
      AT_DISPATCH_V2(
          self.scalar_type(),
          "random_from_to_range_calc",
          AT_WRAP([&] {
            if constexpr (std::is_same_v<scalar_t, bool>) {
              to_inc = static_cast<int64_t>(true);
            } else {
              to_inc =
                  static_cast<int64_t>(std::numeric_limits<scalar_t>::max());
            }
          }),
          AT_EXPAND(AT_INTEGRAL_TYPES_V2),
          kBool);
    } else {
      TORCH_CHECK(
          false,
          "random_from_to_impl handles only integral, floating-point and boolean types");
    }
    native::templates::check_from_to_in_range(from, to_inc, self.dtype());
    CHECK_EMPTY_AND_RETURN(self);
    range = static_cast<uint64_t>(to_inc) - static_cast<uint64_t>(from) + 1;
    native::xpu::random_from_to_kernel(iter, range, from, generator);
  } else {
    // [std::numeric_limits<int64_t>::lowest(),
    // std::numeric_limits<int64_t>::max()] range = 2^64
    CHECK_EMPTY_AND_RETURN(self);
    native::xpu::random_full_64_bits_range_kernel(iter, generator);
  }
  return self;
}

Tensor& XPUNativeFunctions::random_(
    Tensor& self,
    ::std::optional<Generator> generator) {
  CHECK_EMPTY_AND_RETURN(self);
  auto iter = at::TensorIterator::borrowing_nullary_op(self);
  native::xpu::random_kernel(iter, generator);
  return self;
}

} // namespace at
