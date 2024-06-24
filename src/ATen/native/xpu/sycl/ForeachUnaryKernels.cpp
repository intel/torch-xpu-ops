#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

namespace at::native::xpu {

template <typename scalar_t, template <class> class Op>
std::vector<Tensor> foreach_unary_op(TensorList tensors) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  std::vector<at::Tensor> vec_res;
  vec_res.reserve(tensors.size());
  for (const auto& t : tensors) {
    vec_res.emplace_back(at::native::empty_like(t));
  }

  tensor_lists.emplace_back(tensors.vec());
  tensor_lists.emplace_back(std::move(vec_res));

  using opmath_t = typename at::opmath_type<scalar_t>;
  multi_tensor_apply<2>(
      tensor_lists,
      UnaryOpFunctor<
          scalar_t,
          /* depth */ 2,
          /* r_args_depth */ 1,
          /* res_arg_index */ 1>(),
      Op<opmath_t>());

  return tensor_lists[1];
}

template <typename scalar_t, template <class> class Op>
void foreach_unary_op_(TensorList tensors) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());
  using opmath_t = typename at::opmath_type<scalar_t>;
  multi_tensor_apply<1>(
      tensor_lists,
      UnaryOpFunctor<
          scalar_t,
          /* depth */ 1,
          /* r_args_depth */ 1,
          /* res_arg_index */ 0>(),
      Op<opmath_t>());
  increment_version(tensors);
}

template <template <class> class Op>
std::vector<Tensor> floating_complex_half_bfloat16(TensorList tensors) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu",
      [&]() { return foreach_unary_op<scalar_t, Op>(tensors); });
}

template <template <class> class Op>
void floating_complex_half_bfloat16_(TensorList tensors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu_",
      [&]() { foreach_unary_op_<scalar_t, Op>(tensors); });
}

// makes the functor
#define STD_FUNCTOR(op_name, functor_name) \
  template <typename T>                    \
  struct functor_name {                    \
    T operator()(T t) const {              \
      return std::op_name(t);              \
    }                                      \
  };

// given a functor and a "dispatch function", creates the outplace and inplace
// operations
#define OP_CUSTOM_FUNCTOR(function, op_name, functor_name)             \
  std::vector<Tensor> foreach_##op_name##_kernel(TensorList tensors) { \
    return function<functor_name>(tensors);                            \
  }                                                                    \
  void foreach_##op_name##_kernel_(TensorList tensors) {               \
    function##_<functor_name>(tensors);                                \
  }

// creates a functor, outplace version, and inplace version.
#define OP(function, op_name, functor_name) \
  STD_FUNCTOR(op_name, functor_name);       \
  OP_CUSTOM_FUNCTOR(function, op_name, functor_name);

OP(floating_complex_half_bfloat16, sqrt, Sqrt);

} // namespace at::native::xpu
