#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Lerp.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/LerpKernels.h>

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(lerp_kernel_tensor_weight, &xpu::lerp_tensor_kernel);
REGISTER_XPU_DISPATCH(lerp_kernel_scalar_weight, &xpu::lerp_scalar_kernel);

} // namespace native

} // namespace at
