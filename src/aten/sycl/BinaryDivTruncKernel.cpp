#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/TensorIterator.h>

#include <aten/sycl/BinaryInternal.h>
#include <aten/sycl/BinaryKernels.h>
#include <aten/sycl/Loops.h>

#include <comm/XPUMathCompat.h>

namespace at::native::xpu {

template <typename scalar_t, typename accscalar_t>
struct DivTruncScalarFunctor {
  DivTruncScalarFunctor(accscalar_t inv_b) : inv_b_(inv_b) {}

  scalar_t operator()(scalar_t a) const {
    return std::trunc(a * inv_b_);
  }

 private:
  accscalar_t inv_b_;
};

template <typename scalar_t>
struct DivTruncFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return c10::xpu::compat::div_trunc(a, b);
  }
};

void div_trunc_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (isIntegralType(dtype, /*includeBool*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(dtype, "div_trunc_xpu", [&]() {
      gpu_kernel_with_scalars(iter, DivFunctor<scalar_t>());
    });
  } else if (iter.is_cpu_scalar(2)) {
    // optimization for floating-point types: if the second operand is a CPU
    // scalar, compute a * reciprocal(b). Note that this may lose one bit of
    // precision compared to computing the division.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_xpu", [&]() {
          using accscalar_t = at::acc_type<scalar_t, true>;
          auto inv_b = accscalar_t(1.0) / iter.scalar_value<accscalar_t>(2);
          iter.remove_operand(2);
          gpu_kernel(iter, DivTruncScalarFunctor<scalar_t, accscalar_t>(inv_b));
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf, kBFloat16, dtype, "div_trunc_xpu", [&]() {
          gpu_kernel_with_scalars<scalar_t>(iter, DivTruncFunctor<scalar_t>());
        });
  }
}

} // namespace at::native::xpu
