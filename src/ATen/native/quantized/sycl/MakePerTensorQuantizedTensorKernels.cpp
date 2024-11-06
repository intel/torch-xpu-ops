#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/quantized/sycl/MakePerTensorQuantizedTensorKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>

namespace at::native::xpu {

template <typename scalar_t, typename underlying_t>
struct AssignQuantizedTensorFunctor {
  scalar_t operator()(underlying_t value) const {
    return scalar_t(value);
  }
};

void assign_quantized_tensor_kernel(const Tensor& self, Tensor& dst) {
  AT_DISPATCH_QINT_TYPES(
      dst.scalar_type(), "assign_quantized_tensor_xpu", [&]() {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(dst)
                        .add_input(self)
                        .build();
        auto caller = AssignQuantizedTensorFunctor<scalar_t, underlying_t>();
        gpu_kernel(iter, caller);
      });
}

} // namespace at::native::xpu
