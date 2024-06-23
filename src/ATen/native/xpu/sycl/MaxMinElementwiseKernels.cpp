#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct MaximumIntFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::max(a, b);
  }
};

template <>
struct MaximumIntFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a || b;
    ;
  }
};

template <typename scalar_t>
struct MaximumFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::max(a, b);
    }
  }
};

void maximum_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(
        iter, MaximumIntFunctor<bool>());
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "max_elementwise_xpu", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, MaximumIntFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "max_elementwise_xpu",
        [&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MaximumFunctor<scalar_t>());
        });
  }
}

template <typename scalar_t>
struct MinimumIntFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::min(a, b);
  }
};

template <>
struct MinimumIntFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a && b;
    ;
  }
};

template <typename scalar_t>
struct MinimumFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    if (a != a) {
      return a;
    } else if (b != b) {
      return b;
    } else {
      return std::min(a, b);
    }
  }
};

void minimum_kernel(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(
        iter, MinimumIntFunctor<bool>());
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "min_elementwise_xpu", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, MinimumIntFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.dtype(),
        "min_elementwise_xpu",
        [&]() {
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MinimumFunctor<scalar_t>());
        });
  }
}

} // namespace at::native::xpu
