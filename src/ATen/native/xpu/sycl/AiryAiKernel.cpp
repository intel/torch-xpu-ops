#include <ATen/ATen.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/MathExtensions.h>

#include <ATen/native/xpu/sycl/AiryAiKernel.h>

namespace at::native::xpu {
template <typename scalar_t>
struct AiryAiFunctor {
  scalar_t operator()(scalar_t a) const {
    return airy_ai_forward(a);
  }
};

void airy_ai_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES(iter.common_dtype(), "airy_ai_xpu", [&]() {
    gpu_kernel(iter, AiryAiFunctor<scalar_t>());
  });
}

} // namespace at::native::xpu