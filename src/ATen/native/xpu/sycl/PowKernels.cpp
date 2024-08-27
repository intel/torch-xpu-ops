#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Pow.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/UnaryFractionKernels.h>
#include <ATen/native/xpu/sycl/UnaryKernels.h>

namespace at {
namespace native {
namespace xpu {

void pow_tensor_scalar_kernel(
    TensorIteratorBase& iter,
    const Scalar& exp_scalar);

namespace impl {

template <typename Base_type, typename Exp_type>
static inline Base_type pow_(Base_type base, Exp_type exp) {
  return std::pow(base, exp);
}

template <typename T>
static inline std::enable_if_t<std::is_integral<T>::value, T> pow_(
    T base,
    T exp) {
  return at::native::powi(base, exp);
}

template <typename T>
static inline c10::complex<T> pow_(c10::complex<T> base, c10::complex<T> exp) {
  return c10_complex_math::pow(base, exp);
}

} // namespace impl

template <typename scalar_t>
struct PowTensorTensorCastFunctor {
  scalar_t operator()(scalar_t base, scalar_t exp) const {
    using opmath_t = at::opmath_type<scalar_t>;
    return impl::pow_(opmath_t{base}, opmath_t{exp});
  }
};

template <typename scalar_t>
struct PowTensorTensorFunctor {
  scalar_t operator()(scalar_t base, scalar_t exp) const {
    return impl::pow_(base, exp);
  }
};

template <typename scalar_t>
struct PowScalarTensorFunctor {
  scalar_t operator()(scalar_t exp) const {
    return impl::pow_(base_, exp);
  }
  PowScalarTensorFunctor(scalar_t base) : base_(base) {}

 private:
  scalar_t base_;
};

template <typename T>
struct PowScalarTensorFunctor<c10::complex<T>> {
  c10::complex<T> operator()(c10::complex<T> exp) const {
    return std::exp(fct_ * exp);
  }
  PowScalarTensorFunctor(c10::complex<T> base) {
    fct_ = std::log(base);
  }

 private:
  c10::complex<T> fct_;
};

template <>
struct PowScalarTensorFunctor<c10::complex<at::Half>> {
  using opmath_t = at::opmath_type<c10::complex<at::Half>>;
  c10::complex<at::Half> operator()(c10::complex<at::Half> exp) const {
    return std::exp(fct_ * opmath_t{exp});
  }
  PowScalarTensorFunctor(c10::complex<at::Half> base) {
    fct_ = std::log(opmath_t{base});
  }

 private:
  opmath_t fct_;
};

template <typename scalar_t>
void pow_scalar_tensor_impl(TensorIteratorBase& iter, scalar_t base) {
  auto f = PowScalarTensorFunctor<scalar_t>(base);
  gpu_kernel(iter, f);
}

template <typename scalar_t, typename opmath_t>
struct PowChalfTensorScalarFunctor {
  scalar_t operator()(scalar_t base) const {
    return std::pow(opmath_t{base}, exp_);
  }
  PowChalfTensorScalarFunctor(opmath_t exp) : exp_(exp) {}

 private:
  opmath_t exp_;
};

void pow_chalf_tensor_scalar_impl(
    TensorIteratorBase& iter,
    const Scalar& exp_scalar) {
  using scalar_t = c10::complex<at::Half>;
  using opmath_t = at::opmath_type<scalar_t>;
  auto exp = exp_scalar.to<opmath_t>();
  auto f = PowChalfTensorScalarFunctor<scalar_t, opmath_t>(exp);
  gpu_kernel(iter, f);
}

void pow_tensor_tensor_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<at::Half>;
    if (iter.is_cpu_scalar(1)) {
      const auto base = iter.scalar_value<scalar_t>(1);
      iter.remove_operand(1);
      pow_scalar_tensor_impl(iter, base);
    } else if (iter.is_cpu_scalar(2)) {
      const auto exp = iter.scalar_value<scalar_t>(2);
      iter.remove_operand(2);
      pow_chalf_tensor_scalar_impl(iter, exp);
    } else {
      TORCH_INTERNAL_ASSERT(!iter.is_cpu_scalar(1) && !iter.is_cpu_scalar(2));
      auto f = PowTensorTensorCastFunctor<scalar_t>();
      gpu_kernel(iter, f);
    }
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "pow_xpu", [&] {
          if (iter.is_cpu_scalar(1)) {
            const auto base = iter.scalar_value<scalar_t>(1);
            iter.remove_operand(1);
            pow_scalar_tensor_impl(iter, base);
          } else if (iter.is_cpu_scalar(2)) {
            const auto exp = iter.scalar_value<scalar_t>(2);
            iter.remove_operand(2);
            pow_tensor_scalar_kernel(iter, exp);
          } else {
            gpu_kernel(iter, PowTensorTensorFunctor<scalar_t>());
          }
        });
  }
}

template <typename scalar_t>
struct PowImplUnaryFunctor1 {
  scalar_t operator()(scalar_t base) const {
    return base * base;
  }
};

template <typename scalar_t>
struct PowImplUnaryFunctor2 {
  scalar_t operator()(scalar_t base) const {
    return base * base * base;
  }
};

template <typename scalar_t>
struct PowImplUnaryFunctor3 {
  scalar_t operator()(scalar_t base) const {
    return 1.0 / (base * base);
  }
};

template <typename scalar_t>
struct PowScalarTensorFunctor2 {
  scalar_t operator()(scalar_t base) const {
    return impl::pow_(base, exp_);
  }
  PowScalarTensorFunctor2(scalar_t exp) : exp_(exp) {}

 private:
  scalar_t exp_;
};

template <typename Base_type, typename Exp_type>
void pow_tensor_scalar_kernel_impl(TensorIteratorBase& iter, Exp_type exp) {
  const auto d_exp = static_cast<double>(exp);
  // .5 (sqrt), -.5 (rsqrt) and -1 (reciprocal) specializations are handled
  // in pow_tensor_scalar_kernel
  if (d_exp == 2) {
    gpu_kernel(iter, PowImplUnaryFunctor1<Base_type>());
  } else if (d_exp == 3) {
    gpu_kernel(iter, PowImplUnaryFunctor2<Base_type>());
  } else if (d_exp == -2) {
    gpu_kernel(iter, PowImplUnaryFunctor3<Base_type>());
  } else {
    auto f = PowScalarTensorFunctor2<Base_type>(exp);
    gpu_kernel(iter, f);
  }
}

void pow_tensor_scalar_kernel(
    TensorIteratorBase& iter,
    const Scalar& exp_scalar) {
  // Dispatch to fast specialization for sqrt, rsqrt and reciprocal
  if (!exp_scalar.isComplex()) {
    if (exp_scalar.equal(.5)) {
      return sqrt_kernel(iter);
    } else if (exp_scalar.equal(-0.5)) {
      return rsqrt_kernel(iter);
    } else if (exp_scalar.equal(-1.0)) {
      return reciprocal_kernel(iter);
    }
  }
  if (isComplexType(iter.common_dtype()) || exp_scalar.isComplex()) {
    if (iter.common_dtype() == kComplexHalf) {
      pow_chalf_tensor_scalar_impl(iter, exp_scalar);
      return;
    }
    AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "pow_xpu", [&]() {
      const auto exp = exp_scalar.to<scalar_t>();
      gpu_kernel(iter, PowScalarTensorFunctor2<scalar_t>(exp));
    });
  } else if (
      isFloatingType(iter.common_dtype()) || exp_scalar.isIntegral(false)) {
    AT_DISPATCH_ALL_TYPES_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "pow_xpu", [&]() {
          const auto exp = exp_scalar.to<scalar_t>();
          pow_tensor_scalar_kernel_impl<scalar_t>(iter, exp);
        });
  } else {
    TORCH_INTERNAL_ASSERT(
        false,
        "invalid combination of type in Pow function, common dtype: ",
        iter.common_dtype(),
        ", exp is integral? ",
        exp_scalar.isIntegral(false));
  }
}

} // namespace xpu
} // namespace native
} // namespace at
