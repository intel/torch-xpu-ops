#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/DispatchStub.h>
#include <ATen/native/quantized/FakeQuantAffine.h>

#include <ATen/native/quantized/sycl/FakeQuantizeCoreKernels.h>

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(
    fake_quant_tensor_cachemask_tensor_qparams_stub,
    &xpu::_fake_quantize_tensor_cachemask_tensor_qparams_kernel);
REGISTER_XPU_DISPATCH(
    fake_quant_grad_learnable_tensor_stub,
    &xpu::_fake_quantize_grad_learnable_tensor_kernel);
REGISTER_XPU_DISPATCH(
    fake_quant_grad_learnable_channel_stub,
    &xpu::_fake_quantize_grad_learnable_channel_kernel);

} // namespace native
} // namespace at