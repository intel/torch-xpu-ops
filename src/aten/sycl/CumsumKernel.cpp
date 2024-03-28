// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <aten/sycl/ScanKernels.h>
#include <aten/sycl/ScanUtils.h>
#include <comm/Numerics.h>

namespace at::native::xpu {

void launch_cumsum_xpu_kernel(
    const Tensor& out,
    const Tensor& self,
    int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      out.scalar_type(),
      "cumsum_xpu",
      [&]() {
        scan<INCLUSIVE_TYPE, scalar_t, scalar_t>(
            out,
            self,
            dim,
            ScalarConvert<float, scalar_t>::to(0.0),
            std::plus<scalar_t>());
      });
}
} // namespace at::native::xpu
