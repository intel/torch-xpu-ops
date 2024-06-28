#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TensorImpl* resize_impl_xpu_(
    TensorImpl* self,
    IntArrayRef size,
    at::OptionalIntArrayRef stride,
    bool device_guard = true);

}
