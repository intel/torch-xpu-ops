#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Activation.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/xpu/ops/gelu_backward_native.h>
#include <ATen/xpu/ops/gelu_native.h>

#include <ATen/native/xpu/sycl/ActivationGeluKernel.h>
#include <ATen/native/xpu/sycl/ActivationThresholdKernel.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(threshold_stub, xpu::threshold_kernel);
TORCH_IMPL_FUNC(gelu_backward_out_xpu)
(const Tensor& /*grad*/,
 const Tensor& /*self*/,
 c10::string_view approximate,
 const Tensor& /*grad_input*/
) {
  xpu::gelu_backward_kernel(*this, approximate);
}

TORCH_IMPL_FUNC(gelu_out_xpu)
(const Tensor& /*self*/, c10::string_view approximate, const Tensor& /*result*/
) {
  xpu::gelu_kernel(*this, approximate);
}

} // namespace native
} // namespace at
