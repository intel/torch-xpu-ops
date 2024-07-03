#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/PointwiseOps.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/BinaryMiscOpsKernels.h>
#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>
#include <comm/RegisterUtils.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(mse_stub, xpu::mse_kernel);
REGISTER_XPU_DISPATCH(mse_backward_stub, xpu::mse_backward_kernel);
} // namespace native
} // namespace at
