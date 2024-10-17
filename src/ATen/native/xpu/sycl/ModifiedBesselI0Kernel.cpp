#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <c10/core/Scalar.h>
#include <c10/util/complex.h>
#include <comm/XPUMathCompat.h>

#include <ATen/native/xpu/sycl/ModifiedBesselI0Kernel.h>

template <typename scalar_t>
struct ModifiedBesselI0Functor {
  scalar_t operator()(scalar_t a) const {
    return modified_bessel_i0_forward(a);
  }
};

void modified_bessel_i0_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "modified_bessel_i0_xpu", [&]() {
    gpu_kernel(iter, ModifiedBesselI0Functor<scalar_t>());
  });
}

}
