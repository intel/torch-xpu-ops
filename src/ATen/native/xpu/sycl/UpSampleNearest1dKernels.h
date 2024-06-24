#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl//UpSample.h>
namespace at::native {
namespace xpu {

void upsample_nearest1d_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales,
    bool is_exact);

void upsample_nearest1d_backward_kernel(
    Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    std::optional<double> scales,
    bool is_exact);

} // namespace xpu

} // namespace at::native
