#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

void hypot_kernel(TensorIteratorBase& iter);

} // namespace xpu
} // namespace native
} // namespace at
