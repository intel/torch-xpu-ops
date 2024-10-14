#include <ATen/AccumulateType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Normalization.h>
#include <ATen/native/xpu/sycl/RenormKernel.h>

#include <comm/RegisterUtils.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(
    renorm_scale_factor_stub,
    &xpu::renorm_scale_factor_kernel);
}
} // namespace at
