#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Activation.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/TensorIterator.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>

#include <ATen/native/xpu/sycl/ActivationEluKernels.h>
#include <ATen/native/xpu/sycl/ActivationGeluKernel.h>
#include <ATen/native/xpu/sycl/ActivationHardshrinkKernels.h>
#include <ATen/native/xpu/sycl/ActivationHardsigmoidKernels.h>
#include <ATen/native/xpu/sycl/ActivationHardswishKernels.h>
#include <ATen/native/xpu/sycl/ActivationHardtanhKernels.h>
#include <ATen/native/xpu/sycl/ActivationLeakyReluKernels.h>
#include <ATen/native/xpu/sycl/ActivationLogSigmoidKernels.h>
#include <ATen/native/xpu/sycl/ActivationMishKernels.h>
#include <ATen/native/xpu/sycl/ActivationPreluKernels.h>
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
REGISTER_XPU_DISPATCH(mish_stub, &xpu::mish_kernel);
REGISTER_XPU_DISPATCH(mish_backward_stub, &xpu::mish_backward_kernel);
REGISTER_XPU_DISPATCH(
    log_sigmoid_backward_stub,
    &xpu::log_sigmoid_backward_kernel);
REGISTER_XPU_DISPATCH(prelu_stub, &xpu::prelu_kernel);
REGISTER_XPU_DISPATCH(prelu_backward_stub, &xpu::prelu_backward_kernel);
REGISTER_XPU_DISPATCH(hardshrink_stub, &xpu::hardshrink_kernel);

TORCH_IMPL_FUNC(gelu_backward_out_xpu)
(const Tensor& /*grad*/,
 const Tensor& /*self*/,
 std::string_view approximate,
 const Tensor& /*grad_input*/
) {
  xpu::gelu_backward_kernel(*this, approximate);
}

TORCH_IMPL_FUNC(gelu_out_xpu)
(const Tensor& /*self*/, std::string_view approximate, const Tensor& /*result*/
) {
  xpu::gelu_kernel(*this, approximate);
}

std::tuple<Tensor&, Tensor&> log_sigmoid_forward_out_xpu(
    const Tensor& input,
    Tensor& result,
    Tensor& buffer) {
  auto iter =
      TensorIteratorConfig().add_output(result).add_const_input(input).build();
  native::xpu::log_sigmoid_forward_kernel(iter);
  return std::forward_as_tuple(result, buffer);
}

std::tuple<Tensor, Tensor> log_sigmoid_forward_xpu(const Tensor& input) {
  auto result = at::empty_like(input);
  auto buffer = at::empty({0}, input.options());
  log_sigmoid_forward_out_xpu(input, result, buffer);
  return std::forward_as_tuple(result, buffer);
}

Tensor& log_sigmoid_backward_xpu_out(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& buffer,
    Tensor& grad_input) {
  auto iter = TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_const_input(input)
                  .add_const_input(grad_output)
                  .build();
  log_sigmoid_backward_stub(kXPU, iter);
  return grad_input;
}

Tensor log_sigmoid_backward_xpu(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& buffer) {
  auto grad_input = at::empty_like(grad_output);
  // NOTE: buffer is only used by CPU dispatch, we just ignore it here
  auto iter = at::TensorIteratorConfig()
                  .add_output(grad_input)
                  .add_const_input(input)
                  .add_const_input(grad_output)
                  .build();
  log_sigmoid_backward_stub(kXPU, iter);
  return iter.output();
}

} // namespace native
} // namespace at
