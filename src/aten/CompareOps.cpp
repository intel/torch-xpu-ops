#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorIterator.h>
// #include <ATen/xpu/XPUNativeFunctions.h>

#include <aten/sycl/CompareKernels.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(eq_stub, xpu::eq_kernel);
REGISTER_XPU_DISPATCH(ne_stub, xpu::ne_kernel);
REGISTER_XPU_DISPATCH(le_stub, xpu::le_kernel);
REGISTER_XPU_DISPATCH(lt_stub, xpu::lt_kernel);
REGISTER_XPU_DISPATCH(gt_stub, xpu::gt_kernel);
REGISTER_XPU_DISPATCH(ge_stub, xpu::ge_kernel);
// REGISTER_XPU_DISPATCH(isnan_stub, xpu::isnan_out)
} // namespace native

// Tensor XPUNativeFunctions::eq(const Tensor& self, const Tensor& other) {
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::eq_kernel(iter);
//   return iter.output();
// }

// Tensor& XPUNativeFunctions::eq_(Tensor& self, const Tensor& other) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(self, self, other);
//   native::xpu::eq_kernel(iter);
//   return self;
// }

// Tensor& XPUNativeFunctions::eq_out(
//     const Tensor& self,
//     const Tensor& other,
//     Tensor& out) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::eq_kernel(iter);
//   return out;
// }

// Tensor XPUNativeFunctions::eq(const Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::eq_kernel(iter); return iter.output();
// }

// Tensor& XPUNativeFunctions::eq_(Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(self, self,
//   wrapper); native::xpu::eq_kernel(iter); return self;
// }

// Tensor& XPUNativeFunctions::eq_out(
//     const Tensor& self,
//     const Scalar& other,
//     Tensor& out) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::eq_kernel(iter); return out;
// }

// Tensor XPUNativeFunctions::ne(const Tensor& self, const Tensor& other) {
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::ne_kernel(iter);
//   return iter.output();
// }

// Tensor& XPUNativeFunctions::ne_(Tensor& self, const Tensor& other) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(self, self, other);
//   native::xpu::ne_kernel(iter);
//   return self;
// }

// Tensor& XPUNativeFunctions::ne_out(
//     const Tensor& self,
//     const Tensor& other,
//     Tensor& out) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::ne_kernel(iter);
//   return out;
// }

// Tensor XPUNativeFunctions::ne(const Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::ne_kernel(iter); return iter.output();
// }

// Tensor& XPUNativeFunctions::ne_(Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(self, self,
//   wrapper); native::xpu::ne_kernel(iter); return self;
// }

// Tensor& XPUNativeFunctions::ne_out(
//     const Tensor& self,
//     const Scalar& other,
//     Tensor& out) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::ne_kernel(iter); return out;
// }

// Tensor XPUNativeFunctions::lt(const Tensor& self, const Tensor& other) {
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::lt_kernel(iter);
//   return iter.output();
// }

// Tensor& XPUNativeFunctions::lt_(Tensor& self, const Tensor& other) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(self, self, other);
//   native::xpu::lt_kernel(iter);
//   return self;
// }

// Tensor& XPUNativeFunctions::lt_out(
//     const Tensor& self,
//     const Tensor& other,
//     Tensor& out) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::lt_kernel(iter);
//   return out;
// }

// Tensor XPUNativeFunctions::lt(const Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::lt_kernel(iter); return iter.output();
// }

// Tensor& XPUNativeFunctions::lt_(Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(self, self,
//   wrapper); native::xpu::lt_kernel(iter); return self;
// }

// Tensor& XPUNativeFunctions::lt_out(
//     const Tensor& self,
//     const Scalar& other,
//     Tensor& out) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::lt_kernel(iter); return out;
// }

// Tensor XPUNativeFunctions::le(const Tensor& self, const Tensor& other) {
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::le_kernel(iter);
//   return iter.output();
// }

// Tensor& XPUNativeFunctions::le_(Tensor& self, const Tensor& other) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(self, self, other);
//   native::xpu::le_kernel(iter);
//   return self;
// }

// Tensor& XPUNativeFunctions::le_out(
//     const Tensor& self,
//     const Tensor& other,
//     Tensor& out) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::le_kernel(iter);
//   return out;
// }

// Tensor XPUNativeFunctions::le(const Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::le_kernel(iter); return iter.output();
// }

// Tensor& XPUNativeFunctions::le_(Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(self, self,
//   wrapper); native::xpu::le_kernel(iter); return self;
// }

// Tensor& XPUNativeFunctions::le_out(
//     const Tensor& self,
//     const Scalar& other,
//     Tensor& out) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::le_kernel(iter); return out;
// }

// Tensor XPUNativeFunctions::gt(const Tensor& self, const Tensor& other) {
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::gt_kernel(iter);
//   return iter.output();
// }

// Tensor& XPUNativeFunctions::gt_(Tensor& self, const Tensor& other) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(self, self, other);
//   native::xpu::gt_kernel(iter);
//   return self;
// }

// Tensor& XPUNativeFunctions::gt_out(
//     const Tensor& self,
//     const Tensor& other,
//     Tensor& out) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::gt_kernel(iter);
//   return out;
// }

// Tensor XPUNativeFunctions::gt(const Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::gt_kernel(iter); return iter.output();
// }

// Tensor& XPUNativeFunctions::gt_(Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(self, self,
//   wrapper); native::xpu::gt_kernel(iter); return self;
// }

// Tensor& XPUNativeFunctions::gt_out(
//     const Tensor& self,
//     const Scalar& other,
//     Tensor& out) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::gt_kernel(iter); return out;
// }

// Tensor XPUNativeFunctions::ge(const Tensor& self, const Tensor& other) {
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::ge_kernel(iter);
//   return iter.output();
// }

// Tensor& XPUNativeFunctions::ge_(Tensor& self, const Tensor& other) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(self, self, other);
//   native::xpu::ge_kernel(iter);
//   return self;
// }

// Tensor& XPUNativeFunctions::ge_out(
//     const Tensor& self,
//     const Tensor& other,
//     Tensor& out) {
//   TensorIterator iter;
//   iter.build_borrowing_comparison_op(out, self, other);
//   native::xpu::ge_kernel(iter);
//   return out;
// }

// Tensor XPUNativeFunctions::ge(const Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   Tensor out;
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::ge_kernel(iter); return iter.output();
// }

// Tensor& XPUNativeFunctions::ge_(Tensor& self, const Scalar& other) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(self, self,
//   wrapper); native::xpu::ge_kernel(iter); return self;
// }

// Tensor& XPUNativeFunctions::ge_out(
//     const Tensor& self,
//     const Scalar& other,
//     Tensor& out) {
//   auto wrapper = native::wrapped_scalar_tensor(other);
//   TensorIterator iter;
//   iter.build_borrowing_except_last_argument_comparison_op(out, self,
//   wrapper); native::xpu::ge_kernel(iter); return out;
// }

// Tensor XPUNativeFunctions::isnan(const Tensor& self) {
//   return XPUNativeFunctions::ne(self, self);
// }

// Tensor& XPUNativeFunctions::isnan_out(const Tensor& self, Tensor& out) {
//   return XPUNativeFunctions::ne_out(self, self, out);
// }

} // namespace at
