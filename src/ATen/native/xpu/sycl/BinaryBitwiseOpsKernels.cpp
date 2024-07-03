#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/Loops.h>

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

} // namespace xpu
} // namespace native
} // namespace at
