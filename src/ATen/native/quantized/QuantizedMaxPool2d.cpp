#include <ATen/core/Tensor.h>
#include <ATen/native/Pool.h>
#include <ATen/native/quantized/sycl/QuantizedMaxPool2d.h>
#include <ATen/native/utils/ParamUtils.h>
#include <comm/RegisterUtils.h>
#include <torch/library.h>

namespace at {
namespace native {

Tensor quantized_max_pool2d_xpu(
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode) {
  return xpu::quantized_max_pool2d_kernel(
      input, kernel_size, stride, padding, dilation, ceil_mode);
}

// Keep the registry in the anonymous namespace.
namespace {
class QMaxPool_arr_args final {
 public:
  static Tensor run(
      const Tensor& qx,
      std::vector<int64_t> kernel_size,
      std::vector<int64_t> stride,
      std::vector<int64_t> padding,
      std::vector<int64_t> dilation,
      bool ceil_mode) {
    // Now we only support Byte, qint is not supported.
    TORCH_CHECK(
        qx.scalar_type() == c10::ScalarType::Byte,
        "QuantizedMaxPool2d only supports Byte for xpu now");
    return at::native::quantized_max_pool2d_xpu(
        qx, kernel_size, stride, padding, dilation, ceil_mode);
  }
};
} // anonymous namespace

TORCH_LIBRARY_IMPL(quantized, XPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::max_pool2d"),
      TORCH_FN(QMaxPool_arr_args::run));
}
} // namespace native
} // namespace at
