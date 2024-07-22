#include <ATen/autocast_mode.h>

namespace at::native::xpu {

at::Tensor roi_align_autocast_redispatch(
    const at::Tensor& input, // Input feature map.
    const at::Tensor& rois, // List of ROIs to pool over.
    double spatial_scale, // The scale of the image features. ROIs will be
    // scaled to this.
    int64_t pooled_height, // The height of the pooled feature map.
    int64_t pooled_width, // The width of the pooled feature
    int64_t sampling_ratio, // The number of points to sample in each bin
    bool aligned) // The flag for pixel shift
// along each axis.
{
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_align", "")
                       .typed<decltype(roi_align)>();
  return op.call(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

at::Tensor roi_align_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastXPU);
  return roi_align_autocast_redispatch(
             at::autocast::cached_cast(at::kFloat, input),
             at::autocast::cached_cast(at::kFloat, rois),
             spatial_scale,
             pooled_height,
             pooled_width,
             sampling_ratio,
             aligned)
      .to(input.scalar_type());
}

} // namespace at::native::xpu
