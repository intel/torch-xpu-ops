#ifdef USE_C10D_XCCL

#include <c10/util/thread_name.h>
#include <xccl/ProcessGroupXCCL.hpp>
namespace c10d {

HeartbeatMonitorXCCL::HeartbeatMonitorXCCL(ProcessGroupXCCL* pg) {
  pg_ = pg;
  coordCheckIntervalMilSec_ = getCvarInt(TORCH_XCCL_COORD_CHECK_MILSEC, 1000);
  LOG(INFO) << pg_->logPrefix() << "HeartbeatMonitor environments: "
            << "TORCH_XCCL_COORD_CHECK_MILSEC: " << coordCheckIntervalMilSec_;
}

void HeartbeatMonitorXCCL::stop() {
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

void HeartbeatMonitorXCCL::start() {
  TORCH_CHECK(
      !xcclHeartbeatMonitorThread_.joinable(),
      "HeartbeatMonitor thread already started");
  xcclHeartbeatMonitorThread_ =
      std::thread(&HeartbeatMonitorXCCL::runLoop, this);
}

void HeartbeatMonitorXCCL::join() {
  if (xcclHeartbeatMonitorThread_.joinable()) {
    xcclHeartbeatMonitorThread_.join();
    LOG(INFO) << pg_->logPrefix()
              << "ProcessGroupXCCL heart beat monitor thread joined.";
  }
}

void HeartbeatMonitorXCCL::runLoop() {
  c10::setThreadName("pt_xccl_heartbt");

  std::optional<DumpPipe> dumpPipe = std::nullopt;
  // We only need to dump once per PG, so we use local_id_ == 0 for the first PG
  if (pg_->local_id_ == 0) {
    // DumpPipe is one per-trainer process
    dumpPipe.emplace(pg_->getRank());
    while (true) {
      std::unique_lock<std::mutex> lock(monitorMutex_);
      if (monitorWakeUpCV_.wait_for(
              lock, std::chrono::milliseconds(coordCheckIntervalMilSec_), [&] {
                return terminateHeartbeatMonitorThread_.load();
              })) {
        return;
      }
      // Write to pipe files for all ranks to dump debug info
      if (dumpPipe.has_value() && dumpPipe->shouldDump()) {
        LOG(INFO) << pg_->logPrefix()
                  << "Dump signal received through pipe, triggering FR dump.";
        std::future<bool> fut = std::async(std::launch::async, [this]() {
          return this->pg_->dumpDebuggingInfo();
        });
      }
    }
  }
}

} // namespace c10d

#endif // USE_C10D_XCCL
