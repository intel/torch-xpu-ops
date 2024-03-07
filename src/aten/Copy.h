#pragma once

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

namespace at::native::xpu {

Tensor& copy_xpu(Tensor& self, const Tensor& src, bool non_blocking);

}
