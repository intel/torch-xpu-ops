#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

namespace at::native::xpu {

const Tensor& resize_sycl_(
    const Tensor& self,
    IntArrayRef size,
    c10::optional<MemoryFormat> optional_memory_format);

} // at::native::xpu
