#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/Lerp.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/LerpKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct LerpTensorComplexFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t self_val, scalar_t end_val, scalar_t weight_val)
      const {
    opmath_t self_val_f = self_val;
    opmath_t end_val_f = end_val;
    opmath_t weight_val_f = weight_val;
    return lerp(self_val, end_val, weight_val);
  }
};

template <typename scalar_t>
struct LerpTensorFunctor {
  scalar_t operator()(scalar_t self_val, scalar_t end_val, scalar_t weight_val)
      const {
    return lerp(self_val, end_val, weight_val);
  }
};

template <typename scalar_t>
struct LerpScalarComplexFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t self_val, scalar_t end_val) const {
    opmath_t self_val_f = self_val;
    opmath_t end_val_f = end_val;
    return lerp(self_val, end_val, weight_val_);
  }

  LerpScalarComplexFunctor(opmath_t weight_val) : weight_val_(weight_val) {}

 private:
  opmath_t weight_val_;
};

template <typename scalar_t>
struct LerpScalarFunctor {
  using opmath_t = at::opmath_type<scalar_t>;
  scalar_t operator()(scalar_t self_val, scalar_t end_val) const {
    return lerp(self_val, end_val, weight_val_);
  }

  LerpScalarFunctor(opmath_t weight_val) : weight_val_(weight_val) {}

 private:
  opmath_t weight_val_;
};

void lerp_scalar_kernel(
    at::TensorIteratorBase& iter,
    const c10::Scalar& weight);

void lerp_tensor_kernel(at::TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "lerp_xpu", [&] {
      if (iter.is_cpu_scalar(3)) {
        auto weight_val = iter.scalar_value<scalar_t>(3);
        iter.remove_operand(3);
        return lerp_scalar_kernel(iter, weight_val);
      }
      gpu_kernel(iter, LerpTensorComplexFunctor<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "lerp_xpu", [&] {
          if (iter.is_cpu_scalar(3)) {
            auto weight_val = iter.scalar_value<scalar_t>(3);
            iter.remove_operand(3);
            return lerp_scalar_kernel(iter, weight_val);
          }
          gpu_kernel(iter, LerpTensorFunctor<scalar_t>());
        });
  }
}

void lerp_scalar_kernel(
    at::TensorIteratorBase& iter,
    const c10::Scalar& weight) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "lerp_xpu", [&] {
      using opmath_t = at::opmath_type<scalar_t>;
      auto weight_val = weight.to<opmath_t>();
      gpu_kernel(iter, LerpScalarComplexFunctor<scalar_t>(weight_val));
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, dtype, "lerp_xpu", [&] {
          using opmath_t = at::opmath_type<scalar_t>;
          auto weight_val = weight.to<opmath_t>();
          gpu_kernel(iter, LerpScalarFunctor<scalar_t>(weight_val));
        });
  }
}

} // namespace at::native::xpu
