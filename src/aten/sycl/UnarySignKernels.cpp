#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <aten/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SgnFunctor {
  scalar_t operator()(scalar_t a) const {
    using opmath_t = at::opmath_type<scalar_t>;
    if (a == scalar_t(0, 0)) {
      return scalar_t(0, 0);
    } else {
      return a / std::abs(static_cast<opmath_t>(a));
    }
  }
};

template <typename scalar_t>
struct SignFunctor {
  scalar_t operator()(scalar_t a) const {
    return c10::signum(a);
  }
};

template <>
struct SignFunctor<bool> {
  bool operator()(bool a) const {
    return a;
  }
};

void sgn_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, iter.dtype(), "sgn_xpu", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    gpu_kernel(iter, SgnFunctor<opmath_t>());
  });
}

void sign_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (dtype == ScalarType::Bool) {
    gpu_kernel(iter, SignFunctor<bool>());
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, dtype, "sign_xpu", [&] {
          gpu_kernel(iter, SignFunctor<scalar_t>());
        });
  }
}

}
