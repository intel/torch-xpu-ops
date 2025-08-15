#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <ATen/xpu/XPUEvent.h>
#include <xccl/ProcessGroupXCCL.hpp>

namespace c10d {

template <>
float getDurationFromEvent<at::xpu::XPUEvent>(
    at::xpu::XPUEvent& xcclStartEvent,
    at::xpu::XPUEvent& xcclEndEvent) {
  TORCH_CHECK(
      xcclEndEvent.query(),
      "getDuration can only be called after work is succeeded.")
  return xcclStartEvent.elapsed_time(xcclEndEvent);
}

template struct FlightRecorder<at::xpu::XPUEvent>;
} // namespace c10d
#endif // USE_C10D_XCCL
