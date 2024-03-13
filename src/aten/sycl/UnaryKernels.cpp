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
struct AbsFunctor {
  scalar_t operator()(const scalar_t a) const {
    return std::abs(a);
  }
};

template <typename scalar_t>
struct SinFunctor {
  scalar_t operator()(const scalar_t a) const {
    return std::sin(a);
  }
};

template <typename scalar_t>
struct CosFunctor {
  scalar_t operator()(const scalar_t a) const {
    return std::cos(a);
  }
};

template <typename scalar_t>
struct LogFunctor {
  scalar_t operator()(scalar_t a) const {
    return std::log(a);
  }
};

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

template <>
struct RsqrtFunctor<c10::complex<float>> {
  c10::complex<float> operator()(c10::complex<float> a) const {
    return c10::complex<float>(1.0, 0) /
        static_cast<c10::complex<float>>(
               std::sqrt(static_cast<std::complex<float>>(a)));
  }
};

template <>
struct RsqrtFunctor<c10::complex<double>> {
  c10::complex<double> operator()(c10::complex<double> a) const {
    return c10::complex<double>(1.0, 0) /
        static_cast<c10::complex<double>>(
               std::sqrt(static_cast<std::complex<double>>(a)));
  }
};

template <typename scalar_t>
struct TanhFunctor {
  scalar_t operator()(scalar_t a) const {
    using opmath_t = at::opmath_type<scalar_t>;
    return std::tanh(static_cast<opmath_t>(a));
  }
};

template <typename scalar_t>
struct NegFunctor {
  scalar_t operator()(scalar_t a) const {
    return -a;
  }
};

void abs_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs_xpu", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(iter, AbsFunctor<opmath_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(
        ScalarType::Half,
        ScalarType::BFloat16,
        ScalarType::Bool,
        iter.dtype(),
        "abs_xpu",
        [&]() { gpu_kernel(iter, AbsFunctor<scalar_t>()); });
  }
}

void sin_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "sin_xpu", [&]() {
        gpu_kernel(iter, SinFunctor<scalar_t>());
      });
}

void cos_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "cos_xpu", [&]() {
        gpu_kernel(iter, CosFunctor<scalar_t>());
      });
}

void log_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "log_xpu", [&]() {
        gpu_kernel(iter, LogFunctor<scalar_t>());
      });
}

void sqrt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "sqrt_xpu", [&]() {
        gpu_kernel(iter, SqrtFunctor<scalar_t>());
      });
}

void rsqrt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "rsqrt_xpu", [&]() {
        gpu_kernel(iter, RsqrtFunctor<scalar_t>());
      });
}

void tanh_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "tanh_xpu", [&]() {
        gpu_kernel(iter, TanhFunctor<scalar_t>());
      });
}

void neg_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX(iter.dtype(), "neg_xpu", [&]() {
    gpu_kernel(iter, NegFunctor<scalar_t>());
  });
}

} // namespace at::native::xpu
