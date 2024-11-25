#include <ATen/core/Tensor.h>
#include <ATen/native/xpu/sycl/RoiAlignKernels.h>
#include <comm/XPUGuard.h>
#include <comm/xpu_aten.h>
#include <torch/library.h>
namespace at::native::xpu {

at::Tensor roi_align(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(input.is_xpu(), "input must be a XPU tensor");
  TORCH_CHECK(rois.is_xpu(), "rois must be a XPU tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  c10::DeviceGuard device_guard(input.device());
  return roi_align_kernel(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

at::Tensor _roi_align_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(grad.is_xpu(), "grad must be a XPU tensor");
  TORCH_CHECK(rois.is_xpu(), "rois must be a XPU tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  c10::DeviceGuard device_guard(grad.device());

  return roi_align_backward_kernel(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio,
      aligned);
}

} // namespace at::native::xpu
