#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <ATen/native/group_norm.h>
#include <ATen/native/xpu/sycl/GroupNormKernels.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(GroupNormKernel, &xpu::group_norm_kernel);
REGISTER_XPU_DISPATCH(
    GroupNormBackwardKernel,
    &xpu::group_norm_backward_kernel);
} // namespace native
} // namespace at
