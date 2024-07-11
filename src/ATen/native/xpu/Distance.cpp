#include <ATen/native/DispatchStub.h>
#include <ATen/native/Distance.h>
#include <ATen/native/xpu/sycl/DistanceKernels.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(cdist_stub, &xpu::cdist_kernel);
}
} // namespace at
