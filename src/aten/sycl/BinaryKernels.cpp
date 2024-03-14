#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename opmath_t>
struct AddFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a + alpha_ * b;
  }
  AddFunctor(opmath_t alpha) : alpha_(alpha) {}

 private:
  opmath_t alpha_;
};

template <typename opmath_t>
struct MulFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a * b;
  }
};

// Workaround for the error: '*' in boolean context, suggest '&&' instead
// [-Werror=int-in-bool-context]
template <>
struct MulFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a && b;
  }
};

template <typename opmath_t>
struct DivFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a / b;
  }
};

template <typename scalar_t>
struct RemainderIntegralFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    scalar_t r = a % b;
    if (r != 0 && c10::signs_differ(r, b)) {
      r += b;
    }
    return r;
  }
};

template <typename scalar_t>
struct RemainderFloatingFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    auto mod = std::fmod(a, b);
    if (mod != 0 && c10::signs_differ(b, mod)) {
      mod += b;
    }
    return mod;
  }
};

template <typename scalar_t>
struct FmodIntegralFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a % b;
  }
};

template <typename scalar_t>
struct FmodFloatingFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return ::fmod(a, b);
  }
};

void add_kernel(TensorIteratorBase& iter, const c10::Scalar& alpha) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(
        iter, AddFunctor(alpha.to<opmath_t>()));
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "add_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_gpu_kernel_with_scalars<scalar_t>(
              iter, AddFunctor(alpha.to<opmath_t>()));
        });
  }
}

void sub_kernel(TensorIteratorBase& iter, const c10::Scalar& alpha) {
  add_kernel(iter, -alpha);
}

void mul_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, MulFunctor<opmath_t>());
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MulFunctor<opmath_t>());
        });
  }
}

void div_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(iter, DivFunctor<opmath_t>());
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "div_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_gpu_kernel_with_scalars<scalar_t>(
              iter, DivFunctor<opmath_t>());
        });
  }
}

void remainder_kernel(TensorIteratorBase& iter) {
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "remainder_xpu", [&]() {
      gpu_kernel_with_scalars(iter, RemainderIntegralFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "remainder_xpu", [&]() {
          gpu_kernel_with_scalars(iter, RemainderFloatingFunctor<scalar_t>());
        });
  }
}

void fmod_kernel(TensorIteratorBase& iter) {
  if (isIntegralType(iter.common_dtype(), /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.common_dtype(), "fmod_xpu", [&]() {
      gpu_kernel_with_scalars(iter, FmodIntegralFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, iter.common_dtype(), "fmod_xpu", [&]() {
          gpu_kernel_with_scalars(iter, FmodFloatingFunctor<scalar_t>());
        });
  }
}

} // namespace xpu
} // namespace native
} // namespace at
