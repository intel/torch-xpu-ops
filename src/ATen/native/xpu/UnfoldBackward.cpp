#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnfoldBackward.h>
#include <ATen/native/xpu/sycl/UnfoldBackwardKernels.h>
#include <comm/xpu_aten.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(unfold_backward_stub, xpu::unfold_backward_kernel);
}
} // namespace at
