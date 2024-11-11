#include <ATen/native/sparse/xpu/sycl/SparseTensorKernels.h>

namespace at::native {

using namespace at::sparse;

SparseTensor _coalesce_sparse_xpu(const SparseTensor& self) {
  return xpu::coalesce_sparse_kernel(self);
}

} // namespace at::native
