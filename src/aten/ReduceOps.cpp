#include <ATen/ScalarOps.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Fill.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <aten/sycl/Scan.h>
#include <ScanKernels.h>

namespace at {
namespace native {
namespace xpu {

template <class Stub>
void impl_func_cum_ops(
    const Tensor& self,
    int64_t dim,
    const Tensor& result,
    Stub& stub) {
  NoNamesGuard guard;
  if (self.dim() == 0) {
    result.fill_(self);
  } else if (self.numel() == 0) {
    result.zero_();
  } else {
    dim = maybe_wrap_dim(dim, self.dim());
    stub(result, self.to(result.scalar_type()), dim);
  }
}

Tensor& XPUNativeFunctions::cumsum_out(
      const Tensor & self,
      int64_t dim,
      c10::optional<ScalarType> dtype,
      Tensor & out) {
  impl_func_cum_ops(self, dim, out, launch_cumsum_xpu_kernel);
  return out;
}

// Tensor XPUNativeFunctions::cumsum(
//     const Tensor & self,
//     int64_t dim,
//     c10::optional<ScalarType> dtype) {
  
// }

} // xpu
} // native
} // at