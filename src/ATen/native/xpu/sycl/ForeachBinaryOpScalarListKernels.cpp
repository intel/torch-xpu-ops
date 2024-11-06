#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarListKernels.h>
#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <ATen/ops/empty_like_native.h>
namespace at::native::xpu {

template <typename scalar_t, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(vec_res);
  using opmath_t = at::opmath_type<scalar_t>;
  multi_tensor_apply<2, opmath_t>(
      tensor_lists,
      scalars,
      BinaryOpScalarListFunctor<
          scalar_t,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),

      Op<opmath_t>());

  return tensor_lists[1];
}

template <typename scalar_t, template <class> class Op>
void foreach_binary_op_(TensorList tensors, at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  using opmath_t = at::opmath_type<scalar_t>;
  multi_tensor_apply<1, opmath_t>(
      tensor_lists,
      scalars,
      BinaryOpScalarListFunctor<
          scalar_t,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>());
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_xpu",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_xpu_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_xpu",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
void all_types_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  AT_DISPATCH_ALL_TYPES_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalar_xpu_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_half_bfloat16(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_xpu",
      [&]() { return foreach_binary_op<scalar_t, Op>(tensors, scalars); });
}

template <template <class> class Op>
void all_types_complex_half_bfloat16_(
    TensorList tensors,
    at::ArrayRef<Scalar> scalars) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_scalarlist_xpu_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalars); });
}

FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(add) {
  return all_types_complex_bool_half_bfloat16_<std::plus>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_KERNEL(add) {
  return all_types_complex_bool_half_bfloat16<std::plus>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(sub) {
  return all_types_complex_bool_half_bfloat16_<std::minus>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_KERNEL(sub) {
  return all_types_complex_bool_half_bfloat16<std::minus>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(mul) {
  return all_types_complex_bool_half_bfloat16_<std::multiplies>(
      tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_KERNEL(mul) {
  return all_types_complex_bool_half_bfloat16<std::multiplies>(
      tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(div) {
  return all_types_complex_bool_half_bfloat16_<std::divides>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_KERNEL(div) {
  return all_types_complex_bool_half_bfloat16<std::divides>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(clamp_max) {
  return all_types_half_bfloat16_<foreach_internal::minimum>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_KERNEL(clamp_max) {
  return all_types_half_bfloat16<foreach_internal::minimum>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(clamp_min) {
  return all_types_half_bfloat16_<foreach_internal::maximum>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_KERNEL(clamp_min) {
  return all_types_half_bfloat16<foreach_internal::maximum>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(pow) {
  return all_types_complex_half_bfloat16_<power_functor>(tensors, scalars);
}

FOREACH_BINARY_SCALARLIST_KERNEL(pow) {
  return all_types_complex_half_bfloat16<power_functor>(tensors, scalars);
}

} // namespace at::native::xpu
