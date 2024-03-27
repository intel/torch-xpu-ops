// CUDA use this
// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <aten/sycl/ScanUtils.h>
#include <aten/sycl/ScanKernels.h>
// #include "comm/MathReduce.h"
// #include "comm/Numerics.h"
// #include "comm/RegistrationDeclarations.h"

namespace at::native::xpu {

Tensor& launch_cumsum_xpu_kernel(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype,
    Tensor& out) {
  // convert input tensor datatype to handle different input/output datatypes
  // case.
  Tensor self_tensor = self.to(out.scalar_type());
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      self_tensor.scalar_type(),
      "cumsum_xpu",
      [&]() {
        scan<INCLUSIVE_TYPE, scalar_t, scalar_t>(
            out,
            self_tensor,
            dim,
            ScalarConvert<float, scalar_t>::to(0.0),
            // TODO: use std::plus<scalar_t>()?
            AddOp<scalar_t>());
      });
  return out;
}
} // namespace at::native::xpu
