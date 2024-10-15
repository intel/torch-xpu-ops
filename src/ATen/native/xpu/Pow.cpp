#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Pow.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/PowKernels.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(pow_tensor_tensor_stub, &xpu::pow_tensor_tensor_kernel);
REGISTER_XPU_DISPATCH(pow_tensor_scalar_stub, &xpu::pow_tensor_scalar_kernel);
} // namespace native
} // namespace at
