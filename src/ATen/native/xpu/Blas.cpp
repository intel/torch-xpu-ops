#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>

namespace at::native {

TORCH_XPU_API Tensor& mm_complex_out_xpu(
    const at::Tensor& self,
    const at::Tensor& mat2,
    bool is_complex,
    Tensor& result) {
  std::cout << "==== call mm_complex_xpu ====" << std::endl;
  return result;
}

TORCH_XPU_API Tensor mm_complex_xpu(
    const at::Tensor& self,
    const at::Tensor& mat2,
    bool is_complex) {
  Tensor result = at::empty(
      {self.size(0), mat2.size(1)}, self.options().dtype(at::kComplexFloat));
  return mm_complex_out_xpu(self, mat2, true, result);
}

} // namespace at::native

// #include <ATen/native/mkldnn/xpu/Blas.h>
// #include <ATen/native/DispatchStub.h>
// Tensor& mm_complex_out_xpu(
//     const Tensor& self,
//     const Tensor& mat2,
//     Tensor& result) {
//   result = at::empty({0}, self.options());
//   // kernel implementation
//   std::cout << "======== call mm_complex_xpu =============" << std::endl;
//   return result;
// }

// REGISTER_XPU_DISPATCH(mm_complex_stub, &mm_complex_out_xpu);

// // Define the operator
// TORCH_LIBRARY(xpu, m) {
//   m.def("mm_complex(Tensor self, Tensor mat2) -> Tensor");
// }

// // Register the XPU implementation for the operator
// TORCH_LIBRARY_IMPL(xpu, at::kXPU, m) {
//   m.impl("mm_complex_xpu", &mm_complex_kernel_xpu);
// }

// #include <xpu/ATen/ops/mm_native.h>

// Tensor& mm_dtype_out_xpu(
//     const at::Tensor& self,
//     const at::Tensor& mat2,
//     at::ScalarType out_dtype,
//     Tensor& result) {
//   std::cout << "======== call mm_complex_xpu =============" << std::endl;
//   return result;
// }

// Tensor mm_dtype_xpu(
//     const at::Tensor& self,
//     const at::Tensor& mat2,
//     at::ScalarType out_dtype) {
//   Tensor result =
//       at::empty({self.size(0), mat2.size(1)},
//       self.options().dtype(out_dtype));
//   return mm_dtype_out_xpu(self, mat2, out_dtype, result);
// }