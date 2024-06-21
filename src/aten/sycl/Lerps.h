#include <ATen/native/Lerp.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/Loops.h>
#include <ATen/OpMathType.h>

namespace at {
namespace native {
namespace xpu {

void lerp_tensor_kernel(at::TensorIteratorBase& iter);

void lerp_scalar_kernel(
    at::TensorIteratorBase& iter,
    const c10::Scalar& weight);

} // namespace xpu
} // namespace native
} // namespace at
