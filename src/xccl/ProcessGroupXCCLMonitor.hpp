#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#ifdef USE_C10D_XCCL
namespace c10d {

// This definition will later be moved to a common header for ProcessGroups NCCL/Gloo/XCCL
#if defined(__linux__)
struct DumpPipe {
  DumpPipe(int rank) {
    std::string fileStem =
        getCvarString({"TORCH_FR_DEBUG_INFO_PIPE_FILE"}, "");
    if (fileStem.empty() ||
        getCvarInt({"TORCH_FR_BUFFER_SIZE"}, 0) <= 0) {
      return;
    }
    TORCH_CHECK(!fileStem.empty(), "TORCH_FR_DEBUG_INFO_PIPE_FILE is empty");
    std::string filename = c10::str(fileStem, rank, ".pipe");
    TORCH_CHECK(
        unlink(filename.c_str()) != -1 || errno == ENOENT,
        "Error removing existing named pipe ",
        filename,
        ", Error: ",
        std::strerror(errno));
    TORCH_CHECK(
        mkfifo(filename.c_str(), 0666) != -1,
        "Error creating named pipe ",
        filename,
        ", Error: ",
        std::strerror(errno));
    fd_ = open(filename.c_str(), O_RDONLY | O_NONBLOCK);
    LOG(INFO) << "Pipe file " << filename
              << " has been opened, write to it to trigger ProcessGroup Debug Dump.";
    TORCH_CHECK(fd_ != -1, "Error opening named pipe ", filename);
  }
  bool shouldDump() {
    if (fd_ == -1) {
      return false;
    }
    // NOLINTNEXTLINE(*array*)
    char buf[128]{};
    // non-blocking from O_NONBLOCK above.
    // Ignore EINTR because we already will poll this
    // again later.
    ssize_t bytesRead = read(fd_, &buf, 128);
    return bytesRead > 0;
  }
  ~DumpPipe() {
    if (fd_ != -1) {
      close(fd_);
    }
  }

 private:
  int fd_ = -1;
};
#else
struct DumpPipe {
  DumpPipe(int rank) {}
  bool shouldDump() {
    return false;
  }
};
#endif

class ProcessGroupXCCL;
class HeartbeatMonitorXCCL {
 public:
  HeartbeatMonitorXCCL(ProcessGroupXCCL* pg);
  virtual ~HeartbeatMonitorXCCL() = default;

  std::string getXCCLTimeoutErrorMsg(const std::string& extraMsg);
  void start();
  void join();
  virtual void runLoop();
  void stop();

 protected:
  ProcessGroupXCCL* pg_;

 private:
  int coordCheckIntervalMilSec_;
  std::condition_variable monitorWakeUpCV_;
  std::mutex monitorMutex_;
  std::thread xcclHeartbeatMonitorThread_;
  std::atomic<bool> terminateHeartbeatMonitorThread_{false};
};
}
#endif // USE_C10D_XCCL
