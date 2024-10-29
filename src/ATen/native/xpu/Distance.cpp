#include <ATen/native/DispatchStub.h>
#include <ATen/native/Distance.h>
#include <ATen/native/xpu/sycl/DistanceKernels.h>

namespace at {
namespace native {

REGISTER_XPU_DISPATCH(cdist_stub, &xpu::cdist_kernel);
REGISTER_XPU_DISPATCH(cdist_backward_stub, &xpu::cdist_backward_kernel);
REGISTER_XPU_DISPATCH(pdist_forward_stub, &xpu::pdist_forward_kernel);
REGISTER_XPU_DISPATCH(pdist_backward_stub, &xpu::pdist_backward_kernel);

} // namespace native
} // namespace at
