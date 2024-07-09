#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct MSEFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    auto diff = a - b;
    return diff * diff;
  }
};

void mse_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.dtype(),
      "mse_xpu",
      [&]() { gpu_kernel(iter, MSEFunctor<scalar_t>()); });
}

} // namespace at::native::xpu
