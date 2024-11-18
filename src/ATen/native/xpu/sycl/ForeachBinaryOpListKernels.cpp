#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpListKernels.h>
#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <ATen/ops/empty_like_native.h>

namespace at::native::xpu {
template <typename T, template <class> class Op>
std::vector<Tensor> foreach_tensor_list_op(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors1.size());
  for (const auto& t : tensors1) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<3>(
      tensor_lists,
      BinaryOpListAlphaFunctor<
          T,
          /* depth */ 3,
          /* r_args_depth */ 2,
          /* res_arg_index */ 2>(),
      Op<opmath_t>(),
      alpha.to<opmath_t>());

  return tensor_lists[2];
}

template <typename T, template <class> class Op>
void foreach_tensor_list_op_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors1.vec());
  tensor_lists.emplace_back(tensors2.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2>(
      tensor_lists,
      BinaryOpListAlphaFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 2,
          /* res_arg_index */ 0>(),
      Op<opmath_t>(),
      alpha.to<opmath_t>());
  increment_version(tensors1);
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_xpu",
      [&]() {
        return foreach_tensor_list_op<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_xpu_",
      [&]() {
        foreach_tensor_list_op_<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_xpu",
      [&]() {
        return foreach_tensor_list_op<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
void all_types_complex_half_bfloat16_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_xpu_",
      [&]() {
        foreach_tensor_list_op_<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
void all_types_half_bfloat16_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_xpu_",
      [&]() {
        foreach_tensor_list_op_<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_half_bfloat16(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kBFloat16,
      kHalf,
      tensors1[0].scalar_type(),
      "foreach_binary_op_list_xpu",
      [&]() {
        return foreach_tensor_list_op<scalar_t, Op>(tensors1, tensors2, alpha);
      });
}

FOREACH_BINARY_LIST_ALPHA_INPLACE_KERNEL(add) {
  return all_types_complex_bool_half_bfloat16_<std::plus>(
      tensor1, tensor2, alpha);
}

FOREACH_BINARY_LIST_ALPHA_KERNEL(add) {
  return all_types_complex_bool_half_bfloat16<std::plus>(
      tensor1, tensor2, alpha);
}

FOREACH_BINARY_LIST_ALPHA_INPLACE_KERNEL(sub) {
  return all_types_complex_bool_half_bfloat16_<std::minus>(
      tensor1, tensor2, alpha);
}

FOREACH_BINARY_LIST_ALPHA_KERNEL(sub) {
  return all_types_complex_bool_half_bfloat16<std::minus>(
      tensor1, tensor2, alpha);
}

FOREACH_BINARY_LIST_INPLACE_KERNEL(mul) {
  return all_types_complex_bool_half_bfloat16_<std::multiplies>(
      tensor1, tensor2);
}

FOREACH_BINARY_LIST_KERNEL(mul) {
  return all_types_complex_bool_half_bfloat16<std::multiplies>(
      tensor1, tensor2);
}

FOREACH_BINARY_LIST_INPLACE_KERNEL(div) {
  return all_types_complex_bool_half_bfloat16_<std::divides>(tensor1, tensor2);
}

FOREACH_BINARY_LIST_KERNEL(div) {
  return all_types_complex_bool_half_bfloat16<std::divides>(tensor1, tensor2);
}

FOREACH_BINARY_LIST_INPLACE_KERNEL(clamp_max) {
  return all_types_half_bfloat16_<foreach_internal::minimum>(tensor1, tensor2);
}

FOREACH_BINARY_LIST_KERNEL(clamp_max) {
  return all_types_half_bfloat16<foreach_internal::minimum>(tensor1, tensor2);
}

FOREACH_BINARY_LIST_INPLACE_KERNEL(clamp_min) {
  return all_types_half_bfloat16_<foreach_internal::maximum>(tensor1, tensor2);
}

FOREACH_BINARY_LIST_KERNEL(clamp_min) {
  return all_types_half_bfloat16<foreach_internal::maximum>(tensor1, tensor2);
}

FOREACH_BINARY_LIST_INPLACE_KERNEL(pow) {
  return all_types_complex_half_bfloat16_<power_functor>(tensor1, tensor2);
}

FOREACH_BINARY_LIST_KERNEL(pow) {
  return all_types_complex_half_bfloat16<power_functor>(tensor1, tensor2);
}

} // namespace at::native::xpu
