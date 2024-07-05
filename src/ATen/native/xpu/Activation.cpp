#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Activation.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/xpu/ops/gelu_backward_native.h>
#include <ATen/xpu/ops/gelu_native.h>

#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/ActivationEluKernels.h>
#include <ATen/native/xpu/sycl/ActivationGeluKernel.h>
#include <ATen/native/xpu/sycl/ActivationHardsigmoidKernels.h>
#include <ATen/native/xpu/sycl/ActivationHardswishKernels.h>
#include <ATen/native/xpu/sycl/ActivationHardtanhKernels.h>
#include <ATen/native/xpu/sycl/ActivationLeakyReluKernels.h>
#include <ATen/native/xpu/sycl/ActivationSiluKernels.h>

#include <ATen/native/xpu/sycl/ActivationSoftplusKernels.h>
#include <ATen/native/xpu/sycl/ActivationSoftshrinkKernels.h>
#include <ATen/native/xpu/sycl/ActivationThresholdKernel.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(threshold_stub, &xpu::threshold_kernel);
REGISTER_XPU_DISPATCH(elu_stub, &xpu::elu_kernel);
REGISTER_XPU_DISPATCH(elu_backward_stub, &xpu::elu_backward_kernel);
REGISTER_XPU_DISPATCH(silu_stub, &xpu::silu_kernel);
REGISTER_XPU_DISPATCH(silu_backward_stub, &xpu::silu_backward_kernel);
REGISTER_XPU_DISPATCH(hardswish_stub, &xpu::hardswish_kernel);
REGISTER_XPU_DISPATCH(hardswish_backward_stub, &xpu::hardswish_backward_kernel);
REGISTER_XPU_DISPATCH(hardtanh_backward_stub, &xpu::hardtanh_backward_kernel);
REGISTER_XPU_DISPATCH(hardsigmoid_stub, &xpu::hardsigmoid_kernel);
REGISTER_XPU_DISPATCH(
    hardsigmoid_backward_stub,
    &xpu::hardsigmoid_backward_kernel);
REGISTER_XPU_DISPATCH(leaky_relu_stub, &xpu::leaky_relu_kernel);
REGISTER_XPU_DISPATCH(
    leaky_relu_backward_stub,
    &xpu::leaky_relu_backward_kernel);
REGISTER_XPU_DISPATCH(softplus_stub, &xpu::softplus_kernel);
REGISTER_XPU_DISPATCH(softplus_backward_stub, &xpu::softplus_backward_kernel);
REGISTER_XPU_DISPATCH(softshrink_stub, &xpu::softshrink_kernel);
REGISTER_XPU_DISPATCH(shrink_backward_stub, &xpu::softshrink_backward_kernel);

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
