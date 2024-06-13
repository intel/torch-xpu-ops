#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/FillKernel.h>
#include <ATen/xpu/ops/_adaptive_avg_pool2d.h>


namespace at::native {

// Tensor& fill_out(Tensor& self, const Scalar& value) {
//   auto iter = TensorIteratorConfig()
//                   .set_check_mem_overlap(
//                       false) // Fill is idempotent, so overlap is okay
//                   .check_all_same_dtype(false)
//                   .add_output(self)
//                   .resize_outputs(false)
//                   .build();
//   native::xpu::fill_kernel(iter, value);
//   return self;
// }

// Tensor& XPUNativeFunctions::fill_(Tensor& self, const Scalar& value) {
//   return fill_out(self, value);
// }

// Tensor& XPUNativeFunctions::fill_(Tensor& self, const Tensor& value) {
//   TORCH_CHECK(
//       value.dim() == 0,
//       "fill_ only supports 0-dimension value tensor but got tensor with ",
//       value.dim(),
//       " dimensions.");
//   if (self.device() != value.device()) {
//     return fill_out(self, value.item());
//   }
//   // Check if value is a view of self and if it is we clone
//   // it to avoid overwriting self prematurely
//   if (self.is_alias_of(value)) {
//     self.copy_(value.clone());
//   } else {
//     self.copy_(value);
//   }
//   return self;
// }

// Tensor& XPUNativeFunctions::zero_(Tensor& self) {
//   return self.fill_(0);
// }

REGISTER_XPU_DISPATCH(fill_stub, &native::xpu::fill_kernel);

} // namespace at::native
