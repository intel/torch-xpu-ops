#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct BitwiseAndFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a & b;
  }
};

template <>
struct BitwiseAndFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a && b;
  }
};

template <typename scalar_t>
struct BitwiseOrFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a | b;
  }
};

template <>
struct BitwiseOrFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a || b;
  }
};

template <typename scalar_t>
struct BitwiseXorFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a ^ b;
  }
};

template <>
struct BitwiseXorFunctor<bool> {
  bool operator()(bool a, bool b) const {
    return a != b;
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

void bitwise_and_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_and_xpu", [&]() {
    BitwiseAndFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

void bitwise_or_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_or_xpu", [&]() {
    BitwiseOrFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

void bitwise_xor_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_xor_xpu", [&]() {
    BitwiseXorFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
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

} // namespace xpu
} // namespace native
} // namespace at
