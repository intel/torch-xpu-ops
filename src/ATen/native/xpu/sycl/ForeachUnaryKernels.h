#pragma once
#include <ATen/ATen.h>

namespace at::native::xpu {

#define FOREACH_UNARY_INPLACE_KERNEL_NAME(NAME) foreach_tensor_##NAME##__kernel

#define FOREACH_UNARY_KERNEL_NAME(NAME) foreach_tensor_##NAME##_kernel

#define FOREACH_UNARY_INPLACE_KERNEL(NAME) \
  void FOREACH_UNARY_INPLACE_KERNEL_NAME(NAME)(TensorList tensors)

#define FOREACH_UNARY_KERNEL(NAME) \
  std::vector<Tensor> FOREACH_UNARY_KERNEL_NAME(NAME)(TensorList tensors)

FOREACH_UNARY_INPLACE_KERNEL(erf);
FOREACH_UNARY_KERNEL(erf);

FOREACH_UNARY_INPLACE_KERNEL(erfc);
FOREACH_UNARY_KERNEL(erfc);

FOREACH_UNARY_INPLACE_KERNEL(expm1);
FOREACH_UNARY_KERNEL(expm1);

FOREACH_UNARY_INPLACE_KERNEL(lgamma);
FOREACH_UNARY_KERNEL(lgamma);

FOREACH_UNARY_INPLACE_KERNEL(trunc);
FOREACH_UNARY_KERNEL(trunc);

FOREACH_UNARY_INPLACE_KERNEL(floor);
FOREACH_UNARY_KERNEL(floor);

FOREACH_UNARY_INPLACE_KERNEL(ceil);
FOREACH_UNARY_KERNEL(ceil);

FOREACH_UNARY_INPLACE_KERNEL(acos);
FOREACH_UNARY_KERNEL(acos);

FOREACH_UNARY_INPLACE_KERNEL(asin);
FOREACH_UNARY_KERNEL(asin);

FOREACH_UNARY_INPLACE_KERNEL(atan);
FOREACH_UNARY_KERNEL(atan);

FOREACH_UNARY_INPLACE_KERNEL(cosh);
FOREACH_UNARY_KERNEL(cosh);

FOREACH_UNARY_INPLACE_KERNEL(sinh);
FOREACH_UNARY_KERNEL(sinh);

FOREACH_UNARY_INPLACE_KERNEL(tanh);
FOREACH_UNARY_KERNEL(tanh);

FOREACH_UNARY_INPLACE_KERNEL(cos);
FOREACH_UNARY_KERNEL(cos);

FOREACH_UNARY_INPLACE_KERNEL(sin);
FOREACH_UNARY_KERNEL(sin);

FOREACH_UNARY_INPLACE_KERNEL(tan);
FOREACH_UNARY_KERNEL(tan);

FOREACH_UNARY_INPLACE_KERNEL(exp);
FOREACH_UNARY_KERNEL(exp);

FOREACH_UNARY_INPLACE_KERNEL(log);
FOREACH_UNARY_KERNEL(log);

FOREACH_UNARY_INPLACE_KERNEL(log1p);
FOREACH_UNARY_KERNEL(log1p);

FOREACH_UNARY_INPLACE_KERNEL(log2);
FOREACH_UNARY_KERNEL(log2);

FOREACH_UNARY_INPLACE_KERNEL(log10);
FOREACH_UNARY_KERNEL(log10);

FOREACH_UNARY_INPLACE_KERNEL(sqrt);
FOREACH_UNARY_KERNEL(sqrt);

FOREACH_UNARY_INPLACE_KERNEL(sigmoid);
FOREACH_UNARY_KERNEL(sigmoid);

FOREACH_UNARY_INPLACE_KERNEL(round);
FOREACH_UNARY_KERNEL(round);

FOREACH_UNARY_INPLACE_KERNEL(frac);
FOREACH_UNARY_KERNEL(frac);

FOREACH_UNARY_INPLACE_KERNEL(reciprocal);
FOREACH_UNARY_KERNEL(reciprocal);

FOREACH_UNARY_INPLACE_KERNEL(sign);
FOREACH_UNARY_KERNEL(sign);

FOREACH_UNARY_INPLACE_KERNEL(neg);
FOREACH_UNARY_KERNEL(neg);

FOREACH_UNARY_INPLACE_KERNEL(abs);
FOREACH_UNARY_KERNEL(abs);

void foreach_tensor_zero_kernel(TensorList& tensors);

} // namespace at::native::xpu
