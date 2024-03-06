#include <ATen/ATen.h>

#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/Dispatch.h>
#include <c10/core/ScalarType.h>

#include <comm/SYCLContext.h>
#include <aten/sycl/Loops.h>
#include <aten/sycl/CopyKernel.h>

namespace at::native::xpu {

template <typename scalar_t>
struct AbsFunc {
  scalar_t operator()(scalar_t src_val) const {
    return std::abs(src_val);
  }
};

void abs_kernel(TensorIterator& iter) {
     AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        iter.common_dtype(),
        "abs_xpu",
        [&]() {
          gpu_kernel(iter, AbsFunc<scalar_t>());
        });
}

} // namespace at::native::xpu
