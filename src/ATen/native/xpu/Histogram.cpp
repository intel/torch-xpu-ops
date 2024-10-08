#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Histogram.h>
#include <ATen/native/Resize.h>
#include <ATen/native/xpu/sycl/HistogramKernels.h>

namespace at {

namespace native {
REGISTER_XPU_DISPATCH(histogramdd_stub, &xpu::histogramdd_kernel);
REGISTER_XPU_DISPATCH(histogramdd_linear_stub, &xpu::histogramdd_linear_kernel);
REGISTER_XPU_DISPATCH(
    histogram_select_outer_bin_edges_stub,
    &xpu::histogram_select_outer_bin_edges_kernel);

} // namespace native
} // namespace at