/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachPointwiseOpScalarListKernels.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <ATen/ops/empty_like_native.h>

namespace at::native::xpu {
template <template <class> class Op>
std::vector<Tensor> foreach_pointwise_template(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            PointwiseOpScalarFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            Op<opmath_t>(),
            scalar.to<opmath_t>());
      });

  return tensor_lists[3];
}

template <template <class> class Op>
void foreach_pointwise_template_(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& scalar) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op__xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            PointwiseOpScalarFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            Op<opmath_t>(),
            scalar.to<opmath_t>());
      });
  increment_version(input);
}

template <template <class> class Op>
std::vector<Tensor> foreach_pointwise_template(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.reserve(4);
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(input.size());
  for (const auto& t : input) {
    vec_res.emplace_back(at::empty_like(t));
  }

  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op_xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<4, opmath_t>(
            tensor_lists,
            scalars,
            PointwiseOpScalarListFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            Op<opmath_t>());
      });

  return tensor_lists[3];
}

template <template <class> class Op>
void foreach_pointwise_template_(
    TensorList input,
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.reserve(3);
  tensor_lists.emplace_back(input.vec());
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      input[0].scalar_type(),
      "foreach_pointwise_op__xpu",
      [&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        multi_tensor_apply<3, opmath_t>(
            tensor_lists,
            scalars,
            PointwiseOpScalarListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            Op<opmath_t>());
      });
}

// Instantiate templates in kernel sources.
#define INSTANTIATE_FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(NAME, OP)           \
  FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(NAME) {                               \
    return foreach_pointwise_template<OP>(input, tensors1, tensors2, scalars); \
  }                                                                            \
                                                                               \
  FOREACH_POINTWISE_OP_SCALARLIST_INPLACE_KERNEL(NAME) {                       \
    foreach_pointwise_template_<OP>(input, tensors1, tensors2, scalars);       \
  }

INSTANTIATE_FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(addcmul, std::multiplies)
INSTANTIATE_FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(addcdiv, std::divides)

#define INSTANTIATE_FOREACH_POINTWISE_OP_SCALAR_KERNEL(NAME, OP)              \
  FOREACH_POINTWISE_OP_SCALAR_KERNEL(NAME) {                                  \
    return foreach_pointwise_template<OP>(input, tensors1, tensors2, scalar); \
  }                                                                           \
                                                                              \
  FOREACH_POINTWISE_OP_SCALAR_INPLACE_KERNEL(NAME) {                          \
    foreach_pointwise_template_<OP>(input, tensors1, tensors2, scalar);       \
  }

INSTANTIATE_FOREACH_POINTWISE_OP_SCALAR_KERNEL(addcmul, std::multiplies)
INSTANTIATE_FOREACH_POINTWISE_OP_SCALAR_KERNEL(addcdiv, std::divides)

} // namespace at::native::xpu
