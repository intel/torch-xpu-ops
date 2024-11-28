#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/ForeachUnaryKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>
#include <comm/XPUMathCompat.h>
#include <comm/xpu_aten.h>

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
}

template <template <class> class Op>
std::vector<Tensor> floating_complex_half(TensorList tensors) {
  return AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
      ScalarType::Half,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu",
      [&]() { return foreach_unary_op<scalar_t, Op>(tensors); });
}

template <template <class> class Op>
void floating_complex_half_(TensorList tensors) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND1(
      ScalarType::Half,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu_",
      [&]() { foreach_unary_op_<scalar_t, Op>(tensors); });
}

template <template <class> class Op>
std::vector<Tensor> all_types_complex_bfloat16_half_bool(TensorList tensors) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Half,
      ScalarType::BFloat16,
      ScalarType::Bool,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu",
      [&]() { return foreach_unary_op<scalar_t, Op>(tensors); });
}

template <template <class> class Op>
void all_types_complex_bfloat16_half_bool_(TensorList tensors) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Half,
      ScalarType::BFloat16,
      ScalarType::Bool,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu",
      [&]() { foreach_unary_op_<scalar_t, Op>(tensors); });
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

template <template <class> class Op>
std::vector<Tensor> all_types_half_complex_bfloat16(TensorList tensors) {
  return AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu",
      [&]() { return foreach_unary_op<scalar_t, Op>(tensors); });
}

template <template <class> class Op>
void all_types_half_complex_bfloat16_(TensorList tensors) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      at::ScalarType::BFloat16,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu_",
      [&]() { foreach_unary_op_<scalar_t, Op>(tensors); });
}

template <template <class> class Op>
std::vector<Tensor> floating_half(TensorList tensors) {
  return AT_DISPATCH_FLOATING_TYPES_AND(
      ScalarType::Half,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu",
      [&]() { return foreach_unary_op<scalar_t, Op>(tensors); });
}

template <template <class> class Op>
void floating_half_(TensorList tensors) {
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      tensors[0].scalar_type(), "foreach_unary_op_xpu_", [&]() {
        foreach_unary_op_<scalar_t, Op>(tensors);
      });
}

template <template <class> class Op>
std::vector<Tensor> floating_half_bfloat16(TensorList tensors) {
  return AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu",
      [&]() { return foreach_unary_op<scalar_t, Op>(tensors); });
}

template <template <class> class Op>
void floating_half_bfloat16_(TensorList tensors) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      tensors[0].scalar_type(),
      "foreach_unary_op_xpu_",
      [&]() { foreach_unary_op_<scalar_t, Op>(tensors); });
}

#define STD_FUNCTOR(OP_NAME, FUNCTOR_NAME) \
  template <typename T>                    \
  struct FUNCTOR_NAME {                    \
    T operator()(T t) const {              \
      return std::OP_NAME(t);              \
    }                                      \
  }

STD_FUNCTOR(erf, Erf);
STD_FUNCTOR(erfc, Erfc);
STD_FUNCTOR(expm1, Expm1);
STD_FUNCTOR(lgamma, Lgamma);
STD_FUNCTOR(trunc, Truncf);
STD_FUNCTOR(floor, Floor);
STD_FUNCTOR(ceil, Ceil);

FOREACH_UNARY_INPLACE_KERNEL(erf) {
  return floating_half_bfloat16_<Erf>(tensors);
}
FOREACH_UNARY_KERNEL(erf) {
  return floating_half_bfloat16<Erf>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(erfc) {
  return floating_half_bfloat16_<Erfc>(tensors);
}
FOREACH_UNARY_KERNEL(erfc) {
  return floating_half_bfloat16<Erfc>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(expm1) {
  return floating_complex_half_bfloat16_<Expm1>(tensors);
}
FOREACH_UNARY_KERNEL(expm1) {
  return floating_complex_half_bfloat16<Expm1>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(lgamma) {
  return floating_half_bfloat16_<Lgamma>(tensors);
}
FOREACH_UNARY_KERNEL(lgamma) {
  return floating_half_bfloat16<Lgamma>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(trunc) {
  return floating_half_bfloat16_<Truncf>(tensors);
}
FOREACH_UNARY_KERNEL(trunc) {
  return floating_half_bfloat16<Truncf>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(floor) {
  return floating_half_bfloat16_<Floor>(tensors);
}
FOREACH_UNARY_KERNEL(floor) {
  return floating_half_bfloat16<Floor>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(ceil) {
  return floating_half_bfloat16_<Ceil>(tensors);
}
FOREACH_UNARY_KERNEL(ceil) {
  return floating_half_bfloat16<Ceil>(tensors);
}

STD_FUNCTOR(acos, Acos);
STD_FUNCTOR(asin, Asin);
STD_FUNCTOR(atan, Atan);
STD_FUNCTOR(cosh, Cosh);
STD_FUNCTOR(sinh, Sinh);
STD_FUNCTOR(tanh, Tanh);
STD_FUNCTOR(cos, Cos);
STD_FUNCTOR(sin, Sin);
STD_FUNCTOR(tan, Tan);

FOREACH_UNARY_INPLACE_KERNEL(acos) {
  return floating_complex_half_bfloat16_<Acos>(tensors);
}
FOREACH_UNARY_KERNEL(acos) {
  return floating_complex_half_bfloat16<Acos>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(asin) {
  return floating_complex_half_bfloat16_<Asin>(tensors);
}
FOREACH_UNARY_KERNEL(asin) {
  return floating_complex_half_bfloat16<Asin>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(atan) {
  return floating_complex_half_bfloat16_<Atan>(tensors);
}
FOREACH_UNARY_KERNEL(atan) {
  return floating_complex_half_bfloat16<Atan>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(cosh) {
  return floating_complex_half_bfloat16_<Cosh>(tensors);
}
FOREACH_UNARY_KERNEL(cosh) {
  return floating_complex_half_bfloat16<Cosh>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(sinh) {
  return floating_complex_half_bfloat16_<Sinh>(tensors);
}
FOREACH_UNARY_KERNEL(sinh) {
  return floating_complex_half_bfloat16<Sinh>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(tanh) {
  return floating_complex_half_bfloat16_<Tanh>(tensors);
}
FOREACH_UNARY_KERNEL(tanh) {
  return floating_complex_half_bfloat16<Tanh>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(cos) {
  return floating_complex_half_bfloat16_<Cos>(tensors);
}
FOREACH_UNARY_KERNEL(cos) {
  return floating_complex_half_bfloat16<Cos>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(sin) {
  return floating_complex_half_bfloat16_<Sin>(tensors);
}
FOREACH_UNARY_KERNEL(sin) {
  return floating_complex_half_bfloat16<Sin>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(tan) {
  return floating_complex_half_bfloat16_<Tan>(tensors);
}
FOREACH_UNARY_KERNEL(tan) {
  return floating_complex_half_bfloat16<Tan>(tensors);
}

STD_FUNCTOR(exp, Exp);
STD_FUNCTOR(log, Log);
STD_FUNCTOR(log1p, Log1p);
STD_FUNCTOR(log2, Log2);
STD_FUNCTOR(log10, Log10);
STD_FUNCTOR(sqrt, Sqrt);

FOREACH_UNARY_INPLACE_KERNEL(exp) {
  return floating_complex_half_bfloat16_<Exp>(tensors);
}
FOREACH_UNARY_KERNEL(exp) {
  return floating_complex_half_bfloat16<Exp>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(log) {
  return floating_complex_half_bfloat16_<Log>(tensors);
}
FOREACH_UNARY_KERNEL(log) {
  return floating_complex_half_bfloat16<Log>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(log1p) {
  return floating_complex_half_bfloat16_<Log1p>(tensors);
}
FOREACH_UNARY_KERNEL(log1p) {
  return floating_complex_half_bfloat16<Log1p>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(log2) {
  return floating_complex_half_bfloat16_<Log2>(tensors);
}
FOREACH_UNARY_KERNEL(log2) {
  return floating_complex_half_bfloat16<Log2>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(log10) {
  return floating_complex_half_bfloat16_<Log10>(tensors);
}
FOREACH_UNARY_KERNEL(log10) {
  return floating_complex_half_bfloat16<Log10>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(sqrt) {
  return floating_complex_half_bfloat16_<Sqrt>(tensors);
}
FOREACH_UNARY_KERNEL(sqrt) {
  return floating_complex_half_bfloat16<Sqrt>(tensors);
}

template <typename T>
struct Sigmoid {
  T one = T(1);
  T operator()(T t) const {
    return (one / (one + std::exp(-t)));
  }
};

template <typename T>
struct Round {
  T operator()(T t) const {
    return std::nearbyint(t);
  }
};

template <typename T>
struct Trunc {
  T operator()(T t) const {
    return t - std::trunc(t);
  }
};

template <typename T>
struct Reciprocal {
  T one = T(1);
  T operator()(T t) const {
    return (one / t);
  }
};

template <typename T>
struct Sign {
  T operator()(T t) const {
    return c10::signum<T>(t);
  }
};

template <typename T>
struct Rsqrt {
  T operator()(T t) const {
    return c10::xpu::compat::rsqrt(t);
  }
};

template <>
struct Rsqrt<c10::complex<float>> {
  c10::complex<float> operator()(c10::complex<float> t) const {
    const auto one = c10::complex<float>(1.0, 0);
    return one / std::sqrt(t);
  }
};

template <>
struct Rsqrt<c10::complex<double>> {
  c10::complex<double> operator()(c10::complex<double> t) const {
    const auto one = c10::complex<double>(1.0, 0);
    return one / std::sqrt(t);
  }
};

FOREACH_UNARY_INPLACE_KERNEL(sigmoid) {
  return floating_complex_half_bfloat16_<Sigmoid>(tensors);
}
FOREACH_UNARY_KERNEL(sigmoid) {
  return floating_complex_half_bfloat16<Sigmoid>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(round) {
  return floating_half_bfloat16_<Round>(tensors);
}
FOREACH_UNARY_KERNEL(round) {
  return floating_half_bfloat16<Round>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(frac) {
  return floating_half_bfloat16_<Trunc>(tensors);
}
FOREACH_UNARY_KERNEL(frac) {
  return floating_half_bfloat16<Trunc>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(reciprocal) {
  return floating_complex_half_bfloat16_<Reciprocal>(tensors);
}
FOREACH_UNARY_KERNEL(reciprocal) {
  return floating_complex_half_bfloat16<Reciprocal>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(sign) {
  return floating_half_bfloat16_<Sign>(tensors);
}
FOREACH_UNARY_KERNEL(sign) {
  return floating_half_bfloat16<Sign>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(rsqrt) {
  return floating_complex_half_bfloat16_<Rsqrt>(tensors);
}
FOREACH_UNARY_KERNEL(rsqrt) {
  return floating_complex_half_bfloat16<Rsqrt>(tensors);
}

FOREACH_UNARY_INPLACE_KERNEL(neg) {
  return all_types_half_complex_bfloat16_<std::negate>(tensors);
}
FOREACH_UNARY_KERNEL(neg) {
  return all_types_half_complex_bfloat16<std::negate>(tensors);
}

template <typename T>
struct Abs {
  T operator()(T t) const {
    return std::abs(t);
  }
};

FOREACH_UNARY_INPLACE_KERNEL(abs) {
  return all_types_complex_bfloat16_half_bool_<Abs>(tensors);
}
FOREACH_UNARY_KERNEL(abs) {
  return all_types_complex_bfloat16_half_bool<Abs>(tensors);
}

void foreach_tensor_zero_kernel(TensorList& tensors) {
  std::vector<std::vector<at::Tensor>> tensor_lists;
  tensor_lists.emplace_back(tensors.vec());

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      ScalarType::Half,
      ScalarType::BFloat16,
      ScalarType::Bool,
      tensors[0].scalar_type(),
      "foreach_zero_xpu",
      [&]() {
        multi_tensor_apply<1>(
            tensor_lists,
            ZeroFunctor<
                scalar_t,
                /* depth */ 1,
                /* r_args_depth */ 1,
                /* res_arg_index */ 0>());
      });
}

} // namespace at::native::xpu
