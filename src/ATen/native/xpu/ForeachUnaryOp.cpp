#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/ForeachUnaryKernels.h>
#include <ATen/ops/_foreach_abs_native.h>
#include <ATen/ops/_foreach_acos_native.h>
#include <ATen/ops/_foreach_asin_native.h>
#include <ATen/ops/_foreach_atan_native.h>
#include <ATen/ops/_foreach_ceil_native.h>
#include <ATen/ops/_foreach_cos_native.h>
#include <ATen/ops/_foreach_cosh_native.h>
#include <ATen/ops/_foreach_erf_native.h>
#include <ATen/ops/_foreach_erfc_native.h>
#include <ATen/ops/_foreach_exp_native.h>
#include <ATen/ops/_foreach_expm1_native.h>
#include <ATen/ops/_foreach_floor_native.h>
#include <ATen/ops/_foreach_frac_native.h>
#include <ATen/ops/_foreach_lgamma_native.h>
#include <ATen/ops/_foreach_log10_native.h>
#include <ATen/ops/_foreach_log1p_native.h>
#include <ATen/ops/_foreach_log2_native.h>
#include <ATen/ops/_foreach_log_native.h>
#include <ATen/ops/_foreach_neg_native.h>
#include <ATen/ops/_foreach_reciprocal_native.h>
#include <ATen/ops/_foreach_round_native.h>
#include <ATen/ops/_foreach_rsqrt_native.h>
#include <ATen/ops/_foreach_sigmoid_native.h>
#include <ATen/ops/_foreach_sign_native.h>
#include <ATen/ops/_foreach_sin_native.h>
#include <ATen/ops/_foreach_sinh_native.h>
#include <ATen/ops/_foreach_sqrt_native.h>
#include <ATen/ops/_foreach_tan_native.h>
#include <ATen/ops/_foreach_tanh_native.h>
#include <ATen/ops/_foreach_trunc_native.h>
#include <ATen/ops/_foreach_zero_native.h>
#include <ATen/ops/empty_like_native.h>

namespace at {
namespace native {

bool check_complex(at::TensorList tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const auto& t) {
    return at::isComplexType(t.scalar_type());
  });
}

#define FOREACH_UNARY_OP(OP_NAME)                                          \
  std::vector<Tensor> foreach_tensor_##OP_NAME##_xpu(TensorList tensors) { \
    check_foreach_api_restrictions(tensors);                               \
    if (!can_use_fast_route(tensors) ||                                    \
        has_integral_tensor(tensors, /*includeBool */ true)) {             \
      return at::native::foreach_tensor_##OP_NAME##_slow(tensors);         \
    }                                                                      \
    return xpu::FOREACH_UNARY_KERNEL_NAME(OP_NAME)(tensors);               \
  }                                                                        \
  void foreach_tensor_##OP_NAME##_xpu_(TensorList tensors) {               \
    check_foreach_api_restrictions(tensors);                               \
    if (!can_use_fast_route(tensors) ||                                    \
        has_integral_tensor(tensors, /*includeBool */ true)) {             \
      return at::native::foreach_tensor_##OP_NAME##_slow_(tensors);        \
    }                                                                      \
    xpu::FOREACH_UNARY_INPLACE_KERNEL_NAME(OP_NAME)(tensors);              \
  }

FOREACH_UNARY_OP(erf);
FOREACH_UNARY_OP(erfc);
FOREACH_UNARY_OP(expm1);
FOREACH_UNARY_OP(lgamma);
FOREACH_UNARY_OP(trunc);
FOREACH_UNARY_OP(floor);
FOREACH_UNARY_OP(ceil);

FOREACH_UNARY_OP(acos);
FOREACH_UNARY_OP(asin);
FOREACH_UNARY_OP(atan);
FOREACH_UNARY_OP(cosh);
FOREACH_UNARY_OP(sinh);
FOREACH_UNARY_OP(tanh);
FOREACH_UNARY_OP(cos);
FOREACH_UNARY_OP(sin);
FOREACH_UNARY_OP(tan);

FOREACH_UNARY_OP(exp);
FOREACH_UNARY_OP(log);
FOREACH_UNARY_OP(log1p);
FOREACH_UNARY_OP(log2);
FOREACH_UNARY_OP(log10);
FOREACH_UNARY_OP(sqrt);

FOREACH_UNARY_OP(sigmoid);
FOREACH_UNARY_OP(round);
FOREACH_UNARY_OP(frac);
FOREACH_UNARY_OP(reciprocal);
FOREACH_UNARY_OP(sign);
FOREACH_UNARY_OP(rsqrt);

std::vector<Tensor> foreach_tensor_neg_xpu(TensorList tensors) {
  at::native::check_foreach_api_restrictions(tensors);
  if (!can_use_fast_route(tensors)) {
    return at::native::foreach_tensor_neg_slow(tensors);
  }

  TORCH_CHECK(
      tensors[0].scalar_type() != kBool,
      "Negation, the `-` operator, on a bool tensor is not supported. "
      "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  return xpu::FOREACH_UNARY_KERNEL_NAME(neg)(tensors);
}

void foreach_tensor_neg_xpu_(TensorList tensors) {
  check_foreach_api_restrictions(tensors);

  if (!can_use_fast_route(tensors)) {
    return at::native::foreach_tensor_neg_slow_(tensors);
  }

  TORCH_CHECK(
      tensors[0].scalar_type() != kBool,
      "Negation, the `-` operator, on a bool tensor is not supported. "
      "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  xpu::FOREACH_UNARY_INPLACE_KERNEL_NAME(neg)(tensors);
}

std::vector<Tensor> foreach_tensor_abs_xpu(at::TensorList tensors) {
  check_foreach_api_restrictions(tensors);
  const bool has_complex = check_complex(tensors);
  if (!can_use_fast_route(tensors) || has_complex) {
    return at::native::foreach_tensor_abs_slow(tensors);
  }
  return xpu::FOREACH_UNARY_KERNEL_NAME(abs)(tensors);
}

void foreach_tensor_abs_xpu_(at::TensorList tensors) {
  check_foreach_api_restrictions(tensors);
  const bool has_complex = check_complex(tensors);
  if (!can_use_fast_route(tensors) || has_complex) {
    return at::native::foreach_tensor_abs_slow_(tensors);
  }
  xpu::FOREACH_UNARY_INPLACE_KERNEL_NAME(abs)(tensors);
}

void foreach_tensor_zero_xpu_(TensorList tensors) {
  check_foreach_api_restrictions(tensors);

  if (!can_use_fast_route(tensors)) {
    return at::native::foreach_tensor_zero_slow_(tensors);
  }
  xpu::foreach_tensor_zero_kernel(tensors);
}

} // namespace native
} // namespace at
