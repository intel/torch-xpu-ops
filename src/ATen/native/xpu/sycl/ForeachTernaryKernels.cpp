#include <ATen/Dispatch.h>
#include <ATen/native/Lerp.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <ATen/native/xpu/sycl/ForeachTernaryOpListKernels.h>
#include <ATen/native/xpu/sycl/ForeachTernaryOpScalarKernels.h>
#include <ATen/native/xpu/sycl/ForeachTernaryOpScalarListKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LerpFunctor {
  inline scalar_t operator()(
      const scalar_t self,
      const scalar_t end,
      const scalar_t weight) {
    return lerp(self, end, weight);
  }
};

void foreach_lerp_list_kernel(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3,
    TensorList result) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec(), result.vec()};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_lerp_list_xpu",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<4>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 4,
                /* r_args_depth */ 3,
                /* res_arg_index */ 3>(),
            LerpFunctor<opmath_t>());
      });
}

void foreach_lerp_list_kernel_(
    TensorList tensors1,
    TensorList tensors2,
    TensorList tensors3) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), tensors3.vec()};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_lerp_list_xpu_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 3,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>());
      });
}

void foreach_lerp_scalar_kernel(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight,
    TensorList result) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), result.vec()};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_lerp_scalar_xpu",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 2,
                /* res_arg_index */ 2>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });
}

void foreach_lerp_scalar_kernel_(
    TensorList tensors1,
    TensorList tensors2,
    const Scalar& weight) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec()};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_lerp_scalar_xpu_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<2>(
            tensor_lists,
            TernaryOpScalarFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 2,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>(),
            weight.to<opmath_t>());
      });
}

void foreach_lerp_scalarlist_kernel(
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars,
    TensorList result) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec(), result.vec()};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalarlist_xpu",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<3, opmath_t>(
            tensor_lists,
            scalars,
            TernaryOpScalarListFunctor<
                scalar_t,
                /* depth */ 3,
                /* r_args_depth */ 2,
                /* res_arg_index */ 2>(),
            LerpFunctor<opmath_t>());
      });
};

void foreach_lerp_scalarlist_kernel_(
    TensorList tensors1,
    TensorList tensors2,
    at::ArrayRef<Scalar> scalars) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      tensors1.vec(), tensors2.vec()};

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors1[0].scalar_type(),
      "foreach_tensor_lerp_scalarlist_xpu_",
      [&]() {
        using opmath_t = typename at::opmath_type<scalar_t>;
        multi_tensor_apply<2, opmath_t>(
            tensor_lists,
            scalars,
            TernaryOpScalarListFunctor<
                scalar_t,
                /* depth */ 2,
                /* r_args_depth */ 2,
                /* res_arg_index */ 0>(),
            LerpFunctor<opmath_t>());
      });
};

} // namespace at::native::xpu
