#include <ATen/Dispatch_v2.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/BinaryInternal.h>
#include <ATen/native/xpu/sycl/BinaryKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename opmath_t>
struct AddFunctor {
  opmath_t operator()(opmath_t a, opmath_t b) const {
    return a + alpha_ * b;
  }
  AddFunctor(opmath_t alpha) : alpha_(alpha) {}

 private:
  opmath_t alpha_;
};

void add_kernel(TensorIteratorBase& iter, const c10::Scalar& alpha) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(
        iter, AddFunctor(alpha.to<opmath_t>()));
  } else {
    AT_DISPATCH_V2(
        common_dtype,
        "add_xpu",
        AT_WRAP([&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_gpu_kernel_with_scalars<scalar_t>(
              iter, AddFunctor(alpha.to<opmath_t>()));
        }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        kBool,
        kHalf,
        kBFloat16,
        kFloat8_e5m2,
        kFloat8_e4m3fn,
        kFloat8_e5m2fnuz,
        kFloat8_e4m3fnuz);
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
    AT_DISPATCH_V2(
        common_dtype,
        "mul_xpu",
        AT_WRAP([&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MulFunctor<opmath_t>());
        }),
        AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
        kBool,
        kHalf,
        kBFloat16,
        kFloat8_e5m2,
        kFloat8_e4m3fn,
        kFloat8_e5m2fnuz,
        kFloat8_e4m3fnuz);
  }
}

} // namespace at::native::xpu
