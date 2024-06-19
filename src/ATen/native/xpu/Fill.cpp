#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/FillKernel.h>
namespace at::native {
REGISTER_XPU_DISPATCH(fill_stub, &native::xpu::fill_kernel);
} // namespace at::native
