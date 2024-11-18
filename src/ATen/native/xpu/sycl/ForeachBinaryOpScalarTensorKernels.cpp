#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/ForeachBinaryOpScalarTensorKernels.h>
#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <ATen/ops/empty_like_native.h>

namespace at::native::xpu {

template <typename T, template <class> class Op>
std::vector<Tensor> foreach_binary_op(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  TORCH_CHECK(
      scalar.dim() == 0 && scalar.numel() == 1,
      "scalar tensor expected to be 0 dim but it has ",
      scalar.dim(),
      " dimensions and ",
      scalar.numel(),
      " elements.");
  TORCH_CHECK(
      tensors[0].device() == scalar.device(),
      "scalar tensor expected to be on ",
      tensors[0].device(),
      " but is on ",
      scalar.device());
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<2>(
      tensor_lists,
      BinaryOpScalarTensorFunctor<
          T,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),
      Op<opmath_t>(),
      scalar.data_ptr<T>(),
      alpha.to<opmath_t>());
  return tensor_lists[1];
}

template <typename T, template <class> class Op>
void foreach_binary_op_(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  TORCH_CHECK(
      scalar.dim() == 0 && scalar.numel() == 1,
      "scalar tensor expected to be 0 dim but has ",
      scalar.dim(),
      " dimensions and ",
      scalar.numel(),
      " elements.");
  TORCH_CHECK(
      tensors[0].device() == scalar.device(),
      "scalar tensor is expected to be on ",
      tensors[0].device(),
      " but is on ",
      scalar.device());
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  using opmath_t = at::opmath_type<T>;
  multi_tensor_apply<1>(
      tensor_lists,
      BinaryOpScalarTensorFunctor<
          T,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>(),
      scalar.data_ptr<T>(),
      alpha.to<opmath_t>());
  increment_version(tensors);
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bool_half_bfloat16(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_tensor_xpu",
      [&]() {
        return foreach_binary_op<scalar_t, Op>(tensors, scalar, alpha);
      });
}

template <template <class> class Op>
void all_types_complex_bool_half_bfloat16_(
    TensorList tensors,
    const Tensor& scalar,
    const Scalar& alpha = 1) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool,
      kHalf,
      kBFloat16,
      tensors[0].scalar_type(),
      "foreach_binary_op_tensor_xpu_",
      [&]() { foreach_binary_op_<scalar_t, Op>(tensors, scalar, alpha); });
}

FOREACH_BINARY_TENSOR_ALPHA_KERNEL(add) {
  return all_types_complex_bool_half_bfloat16<std::plus>(
      tensors, scalar, alpha);
}

FOREACH_BINARY_TENSOR_ALPHA_INPLACE_KERNEL(add) {
  return all_types_complex_bool_half_bfloat16_<std::plus>(
      tensors, scalar, alpha);
}

FOREACH_BINARY_TENSOR_INPLACE_KERNEL(mul) {
  return all_types_complex_bool_half_bfloat16_<std::multiplies>(
      tensors, scalar);
}

FOREACH_BINARY_TENSOR_KERNEL(mul) {
  return all_types_complex_bool_half_bfloat16<std::multiplies>(tensors, scalar);
}

FOREACH_BINARY_TENSOR_INPLACE_KERNEL(div) {
  return all_types_complex_bool_half_bfloat16_<std::divides>(tensors, scalar);
}

FOREACH_BINARY_TENSOR_KERNEL(div) {
  return all_types_complex_bool_half_bfloat16<std::divides>(tensors, scalar);
}

} // namespace at::native::xpu
