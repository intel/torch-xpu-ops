#include <ATen/ATen.h>
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/LinearAlgebra.h>
#include <ATen/native/utils/ParamUtils.h>

#include <ATen/native/xpu/sycl/LinearAlgebraKernels.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(addr_stub, xpu::addr_kernel);

}
} // namespace at
