#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {
template <typename scalar_t, typename underlying_t>
struct AssignQuantizedTensorFunctor {
  scalar_t operator()(underlying_t value) const {
    return scalar_t(value);
  }
};

void assign_quantized_tensor_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_QINT_TYPES(iter.dtype(), "assign_quantized_tensor_xpu", [&]() {
    auto caller = AssignQuantizedTensorFunctor<scalar_t, underlying_t>();
    gpu_kernel(iter, caller);
  });
}

} // namespace at::native::xpu
