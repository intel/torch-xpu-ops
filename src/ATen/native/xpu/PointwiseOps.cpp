#include <ATen/core/Tensor.h>
#include <ATen/native/PointwiseOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/PointwiseOpsKernels.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(addcmul_stub, xpu::addcmul_kernel);
}
} // namespace at
