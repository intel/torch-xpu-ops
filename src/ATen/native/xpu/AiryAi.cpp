#include <ATen/native/DispatchStub.h>
#include <ATen/native/xpu/sycl/AiryAiKernel.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(airy_ai_stub, &xpu::airy_ai_kernel);

} // namespace native
} // namespace at