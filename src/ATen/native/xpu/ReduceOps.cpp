#include <ATen/ScalarOps.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>

#include <ATen/native/DispatchStub.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ReduceMaxValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceMinValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceOpsKernels.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>
#include <comm/ReduceOpsUtils.h>
#include <torch/library.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(sum_stub, &xpu::sum_kernel);
REGISTER_XPU_DISPATCH(mean_stub, &xpu::mean_kernel);
REGISTER_XPU_DISPATCH(argmax_stub, &xpu::argmax_kernel);
REGISTER_XPU_DISPATCH(and_stub, &xpu::and_kernel);
REGISTER_XPU_DISPATCH(or_stub, &xpu::or_kernel);
REGISTER_XPU_DISPATCH(max_values_stub, &xpu::max_values_kernel);
REGISTER_XPU_DISPATCH(min_values_stub, &xpu::min_values_kernel);
REGISTER_XPU_DISPATCH(std_var_stub, &xpu::std_var_kernel);
} // namespace native
} // namespace at
