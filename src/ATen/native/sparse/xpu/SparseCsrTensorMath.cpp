#include <ATen/native/sparse/xpu/sycl/SparseCsrTensorMathKernels.h>

namespace at::native {

using namespace at::sparse;

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_xpu) (
  const Tensor& input, const int64_t size, const bool out_int32, const Tensor& result){
  xpu::convert_indices_from_coo_to_csr_xpu_kernel(
      input,
      size,
      out_int32,
      result);
}

} // namespace at::native
