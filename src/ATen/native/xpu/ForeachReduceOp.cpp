#include <ATen/native/ForeachUtils.h>

#include <ATen/native/xpu/sycl/ForeachReduceKernels.h>
#include <xpu/ATen/ops/_foreach_max_native.h>
#include <xpu/ATen/ops/_foreach_norm_native.h>

namespace at {
namespace native {

static inline void check_foreach_norm_dtype(
    optional<ScalarType> opt_dtype,
    ScalarType self_dtype,
    const char* const name) {
  if (opt_dtype.has_value()) {
    auto dtype = opt_dtype.value();
    TORCH_CHECK(
        isFloatingType(dtype) || isComplexType(dtype),
        name,
        ": dtype should"
        " be floating point or complex, but got ",
        dtype);
    TORCH_CHECK(
        isComplexType(self_dtype) == isComplexType(dtype),
        name,
        ": dtype should be ",
        isComplexType(self_dtype) ? "complex" : "real",
        " for ",
        isComplexType(self_dtype) ? "complex" : "real",
        " inputs, but got ",
        dtype);
    TORCH_CHECK(
        promoteTypes(self_dtype, dtype) == dtype,
        name,
        ": the dtype of the input ",
        "(",
        self_dtype,
        ") should be convertible ",
        "without narrowing to the specified dtype (",
        dtype,
        ")");
  }
}

std::vector<Tensor> foreach_tensor_norm_xpu(
    TensorList tensors,
    const Scalar& ord,
    c10::optional<ScalarType> dtype) {
  const auto p = [&]() -> double {
    if (ord.isIntegral(false)) {
      return ord.to<int64_t>();
    } else if (ord.isFloatingPoint()) {
      return ord.to<double>();
    } else {
      TORCH_CHECK(false, "foreach_norm_xpu expects ord to be integer or float");
    }
  }();
  at::native::check_foreach_api_restrictions(tensors);
  const bool has_int_or_complex =
      std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
        const auto scalar_type = t.scalar_type();
        return at::isIntegralType(scalar_type, /*includeBool*/ true) ||
            at::isComplexType(scalar_type);
      });
  if (!at::native::can_use_fast_route(tensors) || has_int_or_complex ||
      !(p == static_cast<double>(1) || p == static_cast<double>(2) ||
        p == std::numeric_limits<double>::infinity())) {
    return at::native::foreach_tensor_norm_slow(tensors, ord, dtype);
  }
  check_foreach_norm_dtype(
      dtype, tensors[0].scalar_type(), "_foreach_norm_xpu");

  return native::xpu::foreach_norm_kernel(tensors, ord, p, dtype);
}

std::vector<Tensor> foreach_tensor_max_xpu(TensorList tensors) {
  check_foreach_api_restrictions(tensors);
  if (!can_use_fast_route(tensors)) {
    return foreach_tensor_max_slow(tensors);
  }

  // for parity with max in ReduceAllOps.cpp, as max(empty) is ???
  TORCH_CHECK(
      std::all_of(
          tensors.begin(),
          tensors.end(),
          [](const auto& t) { return t.numel() > 0; }),
      "max(): Expected reduction dim to be specified for input.numel() == 0. Specify the reduction dim with the 'dim' argument.");

  return at::native::xpu::foreach_max_kernel(tensors);
}

} // namespace native
} // namespace at
