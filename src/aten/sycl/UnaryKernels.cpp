#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <aten/sycl/CopyKernel.h>
#include <aten/sycl/Loops.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SqrtFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::sqrt(a);
  }
};

template <typename scalar_t>
struct RsqrtFunctor {
  scalar_t operator()(scalar_t a) const {
    return sycl::rsqrt(float(a));
  }
};

template <>
struct RsqrtFunctor<double> {
  double operator()(double a) const {
    return sycl::rsqrt(a);
  }
};

template <typename T>
struct RsqrtFunctor<c10::complex<T>> {
  c10::complex<T> operator()(c10::complex<T> a) const {
    return c10::complex<T>(1.0, 0) /
        static_cast<c10::complex<T>>(
               std::sqrt(static_cast<std::complex<T>>(a)));
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
struct ExpFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::exp(static_cast<acc_t>(a));
  }
};

template <typename scalar_t>
struct BitwiseNotFunctor {
  scalar_t operator()(scalar_t a) const {
    return ~a;
  }
};

template <>
struct BitwiseNotFunctor<bool> {
  bool operator()(bool a) const {
    return !a;
  }
};

void sqrt_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "sqrt_xpu", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel(iter, SqrtFunctor<opmath_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "sqrt_xpu",
        [&]() { gpu_kernel(iter, SqrtFunctor<scalar_t>()); });
  }
}

void rsqrt_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "rsqrt_xpu", [&]() {
          using opmath_t = at::opmath_type<scalar_t>;
          gpu_kernel(iter, RsqrtFunctor<opmath_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::BFloat16,
        ScalarType::Half,
        iter.common_dtype(),
        "rsqrt_xpu",
        [&]() { gpu_kernel(iter, RsqrtFunctor<scalar_t>()); });
  }
}

void bitwise_not_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, BitwiseNotFunctor<bool>());
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "bitwise_not_xpu", [&]() {
      gpu_kernel(iter, BitwiseNotFunctor<scalar_t>());
    });
  }
}

void exp_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "exp_xpu", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      auto caller = ExpFunctor<scalar_t, opmath_t>();
      gpu_kernel(iter, caller);
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        common_dtype,
        "exp_xpu",
        [&]() {
          auto caller = ExpFunctor<scalar_t>();
          gpu_kernel(iter, caller);
        });
  }
}

} // namespace at::native::xpu
