#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct SgnFunctor {
  scalar_t operator()(scalar_t z_) const {
    using opmath_t = at::opmath_type<scalar_t>;
    opmath_t z = static_cast<opmath_t>(z_);
    if (z == opmath_t(0, 0)) {
      return opmath_t(0, 0);
    } else {
      return z / std::abs(z);
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
    gpu_kernel(iter, SgnFunctor<scalar_t>());
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

template <typename scalar_t>
struct LogicalNotFunctor {
  scalar_t operator()(scalar_t a) const {
    return static_cast<bool>(!a);
  }
};

void logical_not_kernel(TensorIteratorBase& iter) {
  // error check -- this is just ensuring we don't dispatch on types that aren't
  // in ALL_TYPES_AND_COMPLEX_AND3(...) so we don't have to maintain a separate
  // list or to do double dispatch.
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool, kHalf, kBFloat16, iter.dtype(0), "logical_not_xpu", [&]() {});
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kBool, kHalf, kBFloat16, iter.dtype(1), "logical_not_xpu", [&]() {
        gpu_kernel(iter, LogicalNotFunctor<scalar_t>());
      });
}

template <typename scalar_t>
struct NegFunctor {
  scalar_t operator()(scalar_t a) const {
    return -a;
  }
};

void neg_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "neg_xpu", [&]() {
      gpu_kernel(iter, NegFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, dtype, "neg_xpu", [&]() {
          gpu_kernel(iter, NegFunctor<scalar_t>());
        });
  }
}

} // namespace at::native::xpu
