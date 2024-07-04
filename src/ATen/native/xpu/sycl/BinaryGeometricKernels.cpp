#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/xpu/sycl/BinaryInternal.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct Atan2Functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::atan2(a, b);
  }
};

void atan2_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      iter.common_dtype(),
      "atan2_xpu",
      [&]() { gpu_kernel(iter, Atan2Functor<scalar_t>()); });
}

} // namespace at::native::xpu
