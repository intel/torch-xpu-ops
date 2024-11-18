#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/xpu/sycl/SparseBinaryOpIntersectionKernels.h>

namespace at::native {

REGISTER_XPU_DISPATCH(
    mul_sparse_sparse_out_stub,
    &xpu::mul_sparse_sparse_kernel);
REGISTER_XPU_DISPATCH(
    sparse_mask_intersection_out_stub,
    &xpu::sparse_mask_intersection_kernel);
REGISTER_XPU_DISPATCH(
    sparse_mask_projection_out_stub,
    &xpu::sparse_mask_projection_kernel);

} // namespace at::native
