#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/BinaryInternal.h>
#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

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
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "add_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_gpu_kernel_with_scalars<scalar_t>(
              iter, AddFunctor(alpha.to<opmath_t>()));
        });
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
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "mul_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
              iter, MulFunctor<opmath_t>());
        });
  }
}

void div_kernel(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (common_dtype == kComplexHalf) {
    using scalar_t = c10::complex<c10::Half>;
    using opmath_t = opmath_type<scalar_t>;
    opmath_gpu_kernel_with_scalars<scalar_t>(iter, DivFunctor<opmath_t>());
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kHalf, kBFloat16, kBool, iter.common_dtype(), "div_xpu", [&]() {
          using opmath_t = opmath_type<scalar_t>;
          opmath_gpu_kernel_with_scalars<scalar_t>(
              iter, DivFunctor<opmath_t>());
        });
  }
}

template <typename scalar_t>
struct SigmoidBackwardFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    using opmath_t = at::opmath_type<scalar_t>;
    const auto one = opmath_t{1.};
    const auto comp_b = static_cast<opmath_t>(b);
    const auto comp_a = static_cast<opmath_t>(a);
    return static_cast<scalar_t>(comp_a * std::conj((one - comp_b) * comp_b));
  }
};

template <typename scalar_t>
struct SigmoidBackwardFunctor2 {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return a * (scalar_t(1.) - b) * b;
  }
};

void sigmoid_backward_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, dtype, "sigmoid_backward_xpu", [&]() {
          gpu_kernel(iter, SigmoidBackwardFunctor<scalar_t>());
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "sigmoid_backward_cuda",
        [&]() { gpu_kernel(iter, SigmoidBackwardFunctor2<scalar_t>()); });
  }
}

} // namespace xpu
} // namespace native
} // namespace at
