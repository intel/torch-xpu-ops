#include <ATen/TensorIterator.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/IndexKernel.h>

#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/xpu/sycl/TensorTransformationsKernel.h>
#include <comm/xpu_aten.h>

namespace at::native {

REGISTER_XPU_DISPATCH(flip_stub, xpu::flip_kernel);

} // namespace at::native
