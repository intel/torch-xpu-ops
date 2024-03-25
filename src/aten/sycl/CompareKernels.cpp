#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename opmath_t>
struct EqFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a == b;
  }
};

template <typename opmath_t>
struct NeFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a != b;
  }
};

template <typename opmath_t>
struct LtFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a < b;
  }
};

template <typename opmath_t>
struct LeFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a <= b;
  }
};

template <typename opmath_t>
struct GtFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a > b;
  }
};

template <typename opmath_t>
struct GeFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
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
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "eq_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, EqFunctor<opmath_t>());
        });
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
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "ne_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, NeFunctor<opmath_t>());
        });
  }
}

void lt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "lt_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, LtFunctor<opmath_t>());
      });
}

void le_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "le_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, LeFunctor<opmath_t>());
      });
}

void gt_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "gt_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, GtFunctor<opmath_t>());
      });
}

void ge_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "ge_xpu", [&]() {
        using opmath_t = opmath_type<scalar_t>;
        opmath_gpu_kernel_with_scalars<scalar_t>(iter, GeFunctor<opmath_t>());
      });
}

} // namespace xpu
} // namespace native
} // namespace at
