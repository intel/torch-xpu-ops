#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct EqFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a == b;
  }
};

template <typename scalar_t>
struct NeFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a != b;
  }
};

template <typename scalar_t>
struct LtFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a < b;
  }
};

template <typename scalar_t>
struct LeFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a <= b;
  }
};

template <typename scalar_t>
struct GtFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a > b;
  }
};

template <typename scalar_t>
struct GeFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a >= b;
  }
};

void eq_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, EqFunctor<opmath_t>());
  } else {
    AT_DISPATCH_V2(
        common_dtype,
        "eq_xpu",
        AT_WRAP([&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, EqFunctor<scalar_t>());
        }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        kHalf,
        kBFloat16,
        kBool,
        AT_EXPAND(AT_FLOAT8_TYPES),
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void ne_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
        iter, NeFunctor<opmath_t>());
  } else {
    AT_DISPATCH_V2(
        common_dtype,
        "ne_xpu",
        AT_WRAP([&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, NeFunctor<scalar_t>());
        }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        kHalf,
        kBFloat16,
        kBool,
        AT_EXPAND(AT_FLOAT8_TYPES),
        AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  }
}

void lt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "lt_xpu", [&]() {
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, LtFunctor<scalar_t>());
      });
}

void le_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "le_xpu", [&]() {
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, LeFunctor<scalar_t>());
      });
}

void gt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "gt_xpu", [&]() {
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, GtFunctor<scalar_t>());
      });
}

void ge_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "ge_xpu", [&]() {
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, GeFunctor<scalar_t>());
      });
}

} // namespace xpu
} // namespace native
} // namespace at
