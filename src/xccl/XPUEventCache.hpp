#pragma once

#include <array>
#include <deque>
#include <memory>
#include <mutex>

#include <ATen/xpu/XPUEvent.h>
#include <c10/macros/Export.h>

namespace c10d {

class TORCH_API XPUEventCache
    : public std::enable_shared_from_this<XPUEventCache> {
 public:
  XPUEventCache();
  std::shared_ptr<at::xpu::XPUEvent> create(bool timing);
  static std::shared_ptr<XPUEventCache> get(at::DeviceIndex device);

 private:
  std::mutex cacheMutex_;
  // NOTE: We intentionally store raw pointers so that
  // we do not attempt to destroy the event objects on process exit,
  // because cuda may be gone.
  std::array<std::deque<at::xpu::XPUEvent*>, 2>
      eventsArray_; // 0 for timing=false, 1 for timing=true
};

} // namespace c10d
