#pragma once

#include <c10/xpu/XPUFunctions.h>

namespace at::xpu {

// Check if every tensor in a list of tensors matches the current
// device.
inline bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  Device curDevice = Device(kXPU, c10::xpu::current_device());
  for (const Tensor& t : ts) {
    if (t.device() != curDevice)
      return false;
  }
  return true;
}

} // namespace at::xpu
