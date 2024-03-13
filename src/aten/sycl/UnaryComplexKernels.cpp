#include <ATen/ATen.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <aten/sycl/CopyKernel.h>
#include <aten/sycl/Loops.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t>
struct ConjScalarFunc {
  scalar_t operator()(scalar_t src_val) const {
    return std::conj(src_val);
  }
};

void conj_kernel(TensorIterator& iter) {
  AT_DISPATCH_SWITCH(
      iter.common_dtype(),
      "conj_xpu",
      AT_DISPATCH_CASE_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, [&] {
        // Conj is a no-op for non-complex types
        copy_kernel(iter);
      }) AT_DISPATCH_CASE_COMPLEX_TYPES_AND(kComplexHalf, [&] {
        gpu_kernel(iter, ConjScalarFunc<scalar_t>());
      }));
}

template <typename scalar_t>
struct NegConjScalarFunc {
  scalar_t operator()(scalar_t src_val) const {
    return std::conj(-src_val);
  }
};

void neg_conj_kernel(TensorIterator& iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_xpu", [&] {
    gpu_kernel(iter, NegConjScalarFunc<scalar_t>());
  });
}

template <typename scalar_t>
struct NegScalarFunc {
  scalar_t operator()(scalar_t src_val) const {
    return -src_val;
  }
};

void neg_kernel(TensorIterator& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "neg_xpu", [&]() {
      gpu_kernel(iter, NegScalarFunc<scalar_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, dtype, "neg_xpu", [&]() {
          gpu_kernel(iter, NegScalarFunc<scalar_t>());
        });
  }
}

} // namespace at::native::xpu
