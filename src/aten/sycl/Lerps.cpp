#include <ATen/native/Lerp.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>
#include <ATen/OpMathType.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct LerpTensorComplexFunctor {
  scalar_t operator()(scalar_t self_val, scalar_t end_val, scalar_t weight_val) const {
    opmath_t self_val_f = self_val;
    opmath_t end_val_f = end_val;
    opmath_t weight_val_f = weight_val;
    return lerp(self_val, end_val, weight_val);
  }
};

template <typename scalar_t>
struct LerpTensorFunctor {
  scalar_t operator()(scalar_t self_val, scalar_t end_val, scalar_t weight_val) const {
    return lerp(self_val, end_val, weight_val);
  }
};

template <typename scalar_t>
struct LerpScalarComplexFunctor {
  scalar_t operator()(scalar_t self_val, scalar_t end_val) const {
    opmath_t self_val_f = self_val;
    opmath_t end_val_f = end_val;
    return lerp(self_val, end_val, weight_val);
  }
};

template <typename scalar_t>
struct LerpScalarFunctor {
  scalar_t operator()(scalar_t self_val, scalar_t end_val) const {
    return lerp(self_val, end_val, weight_val);
  }
};

void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if(at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf,
        dtype,
        "lerp_xpu",
        [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      at::native::gpu_kernel(iter, LerpTensorComplexFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "lerp_xpu",
        [&] {
      at::native::gpu_kernel(iter, LerpTensorFunctor<scalar_t>());
    });
  }
}

void lerp_scalar_kernel(at::TensorIteratorBase& iter, const c10::Scalar& weight) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, 
        dtype,
        "lerp_xpu",
        [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      auto weight_val = weight.to<opmath_t>();
      at::native::gpu_kernel(iter,LerpScalarComplexFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        dtype,
        "lerp_xpu",
        [&]{
      using opmath_t = at::opmath_type<scalar_t>;
      auto weight_val = weight.to<opmath_t>();
      at::native::gpu_kernel(iter, LerpScalarFunctor<scalar_t>());
    });
  }
}

} // namespace xpu
} // namespace native
} // namespace at
