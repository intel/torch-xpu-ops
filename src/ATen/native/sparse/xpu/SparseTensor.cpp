#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/xpu/sycl/SparseTensorKernels.h>

namespace at::native {

using namespace at::sparse;

SparseTensor _coalesce_sparse_xpu(const SparseTensor& self) {
  return xpu::coalesce_sparse_kernel(self);
}

REGISTER_XPU_DISPATCH(flatten_indices_stub, &xpu::flatten_indices_kernel);

} // namespace at::native
