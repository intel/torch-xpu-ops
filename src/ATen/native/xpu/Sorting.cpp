
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Sorting.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Sorting.h>

#include <ATen/native/ReduceOpsUtils.h>
#include <comm/TensorInfo.h>

#include <comm/RegisterUtils.h>
#include <comm/xpu_aten.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(sort_stub, xpu::sort_stable_kernel);
}
} // namespace at