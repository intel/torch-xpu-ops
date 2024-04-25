#pragma once

#include <ATen/native/TensorIterator.h>

namespace at {
namespace native {
namespace xpu {

Tensor nms_kernel(const Tensor& dets_sorted, float iou_threshold);

}
} // namespace native
} // namespace at
