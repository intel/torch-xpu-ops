#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
// #include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/UnaryKernels.h>
#include <ATen/native/xpu/sycl/UnaryLogKernels.h>
#include <ATen/native/xpu/sycl/UnarySignKernels.h>
#include <ATen/native/xpu/sycl/UnarySpecialOpsKernels.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/real.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(abs_stub, native::xpu::abs_kernel);
REGISTER_XPU_DISPATCH(sin_stub, native::xpu::sin_kernel);
REGISTER_XPU_DISPATCH(cos_stub, native::xpu::cos_kernel);
REGISTER_XPU_DISPATCH(log_stub, native::xpu::log_kernel);
REGISTER_XPU_DISPATCH(sqrt_stub, native::xpu::sqrt_kernel);
REGISTER_XPU_DISPATCH(rsqrt_stub, native::xpu::rsqrt_kernel);
REGISTER_XPU_DISPATCH(tanh_stub, native::xpu::tanh_kernel);
REGISTER_XPU_DISPATCH(neg_stub, native::xpu::neg_kernel);
REGISTER_XPU_DISPATCH(reciprocal_stub, native::xpu::reciprocal_kernel);
REGISTER_XPU_DISPATCH(bitwise_not_stub, native::xpu::bitwise_not_kernel);
REGISTER_XPU_DISPATCH(exp_stub, native::xpu::exp_kernel);
REGISTER_XPU_DISPATCH(sigmoid_stub, native::xpu::sigmoid_kernel);
REGISTER_XPU_DISPATCH(
    sgn_stub,
    native::xpu::sgn_kernel); // how to handle comple
} // namespace native

template <typename Stub>
static inline Tensor& unary_op_impl_out(
    Tensor& result,
    const Tensor& self,
    Stub& stub) {
  auto iter = TensorIterator::unary_op(result, self);
  stub(iter);
  return result;
}

template <typename Stub, typename... Args>
static inline Tensor& unary_op_impl_float_out(
    Tensor& result,
    const Tensor& self,
    Stub& stub,
    Args... args) {
  auto iter = TensorIterator::unary_float_op(result, self);
  stub(iter, args...);
  iter.cast_outputs();
  return result;
}

template <typename Stub>
static inline Tensor& unary_op_impl_with_complex_to_float_out(
    Tensor& result,
    const Tensor& self,
    Stub& stub,
    bool promotes_integer_to_float) {
  if (self.is_complex() && !result.is_complex()) {
    // Checks if the corresponding float type can be cast to the desired dtype
    const auto float_type = c10::toRealValueType(self.scalar_type());
    TORCH_CHECK(
        canCast(float_type, result.scalar_type()),
        "result type ",
        float_type,
        " can't be cast to the desired output type ",
        result.scalar_type());

    // Runs the function complex->complex, as TensorIterator expects
    Tensor complex_result = at::empty({0}, self.options());
    auto iter = TensorIterator::unary_op(complex_result, self);
    stub(iter);

    // Copies the complex result to the actual result and returns it
    at::native::resize_output(result, complex_result.sizes());
    result.copy_(at::real(complex_result));
    return result;
  }

  if (promotes_integer_to_float) {
    return unary_op_impl_float_out(result, self, stub);
  }

  return unary_op_impl_out(result, self, stub);
}

// out_impl passed into unary_op_impl and unary_op_impl_  must go through at::
// device dispatch otherwise it won't dispatch to out-of-source devices like
// XLA. For example it must be at::bitwise_not_out instead of
// bitwise_not_out(which is at::native!).
template <typename OutImpl>
static inline Tensor unary_op_impl(const Tensor& self, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

// An alternate version of unary_op_impl that follows the same pattern
// for non-complex inputs, but returns a floating point tensor
// for complex inputs by default.
template <typename OutImpl>
static inline Tensor unary_op_impl_with_complex_to_float(
    const Tensor& self,
    OutImpl& out_impl) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty_like(self, self.options().dtype(float_type));
    return out_impl(result, self);
  }

  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

template <typename OutImpl>
static inline Tensor& unary_op_impl_(Tensor& self, OutImpl& out_impl) {
  return out_impl(self, self);
}
} // namespace at
