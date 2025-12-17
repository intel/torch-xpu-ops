#include <ATen/native/nested/NestedTensorBinaryOps.h>

#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <ATen/native/xpu/sycl/IndexUtils.h>
#include <ATen/native/xpu/sycl/KernelUtils.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/xpu/XPUContext.h>

#include <comm/XPUMathCompat.h>

#include <ATen/native/nested/NestedTensorBinaryOps.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/nested/xpu/sycl/NestedTensorBinaryOpsKernels.h>

namespace at::native {
REGISTER_XPU_DISPATCH(
    nested_dense_elementwise_stub,
    &xpu::_nested_op_dense_esuhm_xpu)
}
