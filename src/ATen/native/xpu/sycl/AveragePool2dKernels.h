#include <comm/xpu_aten.h>

namespace at::native::xpu {

TORCH_XPU_API void avg_pool2d_kernel(
    const Tensor& input_,
    int64_t kH_,
    int64_t kW_,
    int64_t dH_,
    int64_t dW_,
    int64_t padH_,
    int64_t padW_,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    const Tensor& output);

TORCH_XPU_API void avg_pool2d_backward_kernel(
    const Tensor& gradOutput_,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad,
    c10::optional<int64_t> divisor_override,
    const Tensor& gradInput);

} // namespace at::native::xpu
