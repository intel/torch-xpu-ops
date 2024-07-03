#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/BinaryOps.h>
#include <ATen/native/xpu/sycl/BinaryInternal.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LogicalAndFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return a && b;
  }
};

void logical_and_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBool, ScalarType::BFloat16, dtype, "logical_and_xpu", [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
            iter, LogicalAndFunctor<scalar_t>());
      });
}

template <typename scalar_t>
struct LogicalOrFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return a || b;
  }
};

void logical_or_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBool, ScalarType::BFloat16, dtype, "logical_or_xpu", [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
            iter, LogicalOrFunctor<scalar_t>());
      });
}

template <typename scalar_t>
struct LogicalXorFunctor {
  bool operator()(scalar_t a, scalar_t b) const {
    return bool(a) != bool(b);
  }
};

void logical_xor_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  AT_DISPATCH_ALL_TYPES_AND3(
      kHalf, kBool, ScalarType::BFloat16, dtype, "logical_xor_xpu", [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
            iter, LogicalXorFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu