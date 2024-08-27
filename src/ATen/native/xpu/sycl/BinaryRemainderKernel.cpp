#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

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
  scalar_t operator()(scalar_t a, scalar_t b) const
      __ubsan_ignore_float_divide_by_zero__ {
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
  scalar_t operator()(scalar_t a, scalar_t b) const
      __ubsan_ignore_float_divide_by_zero__ {
    return std::fmod(a, b);
  }
};

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
