#include <ATen/ATen.h>
// #include <ATen/Tensor.h>
#include <aten/sycl/UpSample.h>
namespace at ::native {
namespace xpu {

void upsample_nearest1d_out_kernel(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    c10::optional<double> scales,
    bool is_exact);
}

} // namespace at::native