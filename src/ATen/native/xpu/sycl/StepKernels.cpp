#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t>
struct NextafterFunctor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return std::nextafter(a, b);
  }
};

void nextafter_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16, iter.common_dtype(), "nextafter_xpu", [&]() {
        gpu_kernel_with_scalars(iter, NextafterFunctor<scalar_t>());
      });
}

} // namespace at::native::xpu