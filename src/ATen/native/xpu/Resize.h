#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

namespace at {
namespace native::xpu {

TensorImpl* resize_impl_xpu_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool device_guard = true);

}
} // namespace at
