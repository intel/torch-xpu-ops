#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#ifdef USE_C10D_XCCL
namespace c10d {

// Default environment variable keys
static std::vector<std::string> HEARTBEAT_TIMEOUT_SEC = {"TORCH_PG_HEARTBEAT_TIMEOUT_SEC", "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"};
static std::vector<std::string> WAIT_TIMEOUT_DUMP_MILSEC = {"TORCH_PG_WAIT_TIMEOUT_DUMP_MILSEC", "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"};
static std::vector<std::string> COORD_CHECK_MILSEC = {"TORCH_PG_COORD_CHECK_MILSEC", "TORCH_NCCL_COORD_CHECK_MILSEC"};
static std::vector<std::string> DUMP_ON_TIMEOUT = {"TORCH_PG_DUMP_ON_TIMEOUT", "TORCH_NCCL_DUMP_ON_TIMEOUT"};
static std::vector<std::string> LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN = {"TORCH_PG_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN", "TORCH_NCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN"};
static std::vector<std::string> ENABLE_MONITORING = {"TORCH_PG_ENABLE_MONITORING", "TORCH_NCCL_ENABLE_MONITORING"};
static std::vector<std::string> EXTRA_DUMP_ON_EXEC = {"TORCH_PG_EXTRA_DUMP_ON_EXEC", "TORCH_NCCL_EXTRA_DUMP_ON_EXEC"};
static std::vector<std::string> RETHROW_BACKEND_ERRORS = {"TORCH_PG_RETHROW_BACKEND_ERRORS", "TORCH_NCCL_RETHROW_CUDA_ERRORS"};
static std::vector<std::string> PROPAGATE_ERROR = {"TORCH_PG_PROPAGATE_ERROR", "TORCH_NCCL_PROPAGATE_ERROR"};
static std::vector<std::string> DESYNC_DEBUG = {"TORCH_PG_DESYNC_DEBUG", "TORCH_NCCL_DESYNC_DEBUG"};

constexpr const char* kStoreDumpKey = "exception_dump";
constexpr const char* kStoreErrorSignalKey = "remote_error";
constexpr const int kWorkStatusUpdatePeriodMs = 30 * 1000; // 30 seconds

// This definition will later be moved to a common header for ProcessGroups
// NCCL/Gloo/XCCL
#if defined(__linux__)
struct DumpPipe {
  DumpPipe(int rank) {
    std::string fileStem = getCvarString({"TORCH_FR_DEBUG_INFO_PIPE_FILE"}, "");
    if (fileStem.empty() || getCvarInt({"TORCH_FR_BUFFER_SIZE"}, 0) <= 0) {
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
    LOG(INFO)
        << "Pipe file " << filename
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

// Forward declaration
class ProcessGroupXCCL;

class ProcessGroupInterface : public Backend {
 public:
  class WorkInterface : public Work {
   public:
    WorkInterface(int rank,
                  OpType opType,
                  uint64_t seq,
                  const char* profilingTitle = nullptr,
                  const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt,
                  bool enableTiming = true);
    WorkInterface(const WorkInterface&);
    virtual ~WorkInterface() = default;
    bool checkTimeout(std::optional<std::chrono::milliseconds> timeout = std::nullopt);
    const std::string& logPrefix() const;
    virtual bool isStarted() const = 0;
    virtual void printTraceback() const = 0;
    void setException(std::exception_ptr exception_ptr);

    // Public getter for operation type
    OpType getOpType() const { return opType_; }

    friend std::ostream& operator<<(
        std::ostream& output,
        const WorkInterface& work);

    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;
    c10::intrusive_ptr<at::ivalue::Future> futureWorkResult_;
    uint64_t seq_;
    std::optional<uint64_t> trace_id_;
    std::optional<uint64_t> trace_reset_epoch_;
    bool startTraceUpdated_{false};
    std::chrono::milliseconds opTimeout_{};
    bool timingEnabled_;
    size_t numelIn_ = 0;
    size_t numelOut_ = 0;

   protected:
    // Print the traceback of the collective at call time
    template <typename EventType> std::string getTraceback() const;
    template <typename EventType> void printTraceback() const;
  };

  class HeartbeatMonitor {
   public:
    HeartbeatMonitor(ProcessGroupInterface* pg);
    virtual ~HeartbeatMonitor() = default;

    void start();
    void join();
    void runLoop();
    void stop();
    std::string getWatchdogTimeoutErrorMsg(const std::string& extraMsg);
    std::string getWatchdogTimeoutExitMsg(const std::string& exitReason);
    int getDumpTimeout() const;
    void setLastWorkListUpdateTime() {
      lastWorkListUpdateTime_ = std::chrono::steady_clock::now();
    }

   protected:
    ProcessGroupInterface* pg_;

   private:
    bool logCppStackOnUncleanShutdown_;
    int heartbeatTimeoutInSec_;
    int waitTimeoutDumpInMilSec_;
    int coordCheckIntervalMilSec_;
    bool watchdogHeartbeatMonitorEnabled_;
    std::thread heartbeatMonitorThread_;
    std::atomic<bool> terminateHeartbeatMonitorThread_{false};
    std::condition_variable monitorWakeUpCV_;
    bool dumpOnTimeoutOrEx_;
    std::mutex monitorMutex_;
    std::chrono::time_point<std::chrono::steady_clock> lastWorkListUpdateTime_;
  };

  class DesyncDebugger {
   public:
    DesyncDebugger() = default;

    void init(ProcessGroupInterface* pg);

    void run();

    void logWorkStart(WorkInterface& work);
    void logWorkEnd(WorkInterface& work);

   private:
    bool enabled_{false};
    int rank_ = -1;
    int size_ = -1;
    int globalRank_ = -1;
    int pgId_ = -1;
    std::string pgBackend_;
    c10::intrusive_ptr<Store> store_ = nullptr;
    std::string traceKeyStart_;
    std::string traceKeyEnd_;
  };

  class Watchdog {
   public:
    explicit Watchdog(ProcessGroupInterface* pg);
    virtual ~Watchdog() = default;

    void start();
    void join();
    void run();
    virtual void runLoop() = 0;
    virtual void processWorkList() = 0;
    void notify();
    void checkAndSetRemoteError();
    int getSignalSrcRank(
        c10::intrusive_ptr<Store>& store,
        const std::string& signal);
    uint64_t getHeartbeat() const;
    void setDesyncDebug(bool desyncDebug);

   protected:
    std::string backendErrorString_;
    std::thread watchdogThread_;
    ProcessGroupInterface* pg_;
    std::exception_ptr watchDogException_ = nullptr;
    std::condition_variable watchdogCV_;
    std::atomic_uint64_t heartbeat_;
    bool propagatePgError_;
    bool desyncDebug_ = false;
    DesyncDebugger desyncDebugger_;
    ErrorType error_{ErrorType::SUCCESS};
    bool rethrowBackendErrors_ = false;
  };

 public:
  ProcessGroupInterface(int rank, int size, c10::intrusive_ptr<Store> store);
  virtual ~ProcessGroupInterface() = default;

  // Basic process group information
  const std::string& logPrefix() const;
  const int& globalRank() const;
  uint64_t getUid();
  const c10::intrusive_ptr<Store> globalStore() const;
  virtual std::string getBackendCclVersion() = 0;
  virtual std::chrono::milliseconds getOptionsTimeout() const = 0;

  // Watchdog interface
  uint64_t getWatchdogHeartbeat() const;

  // Error handling
  void terminateProcess(const std::string& errMsg);
  bool dumpDebuggingInfo(bool includeStackTrace = true, bool onlyActive = false);
  void dumpExtraDebuggingInfo();
  void broadcastDumpSignal();
  void broadcastSignal(c10::intrusive_ptr<Store>& store, const std::string& key, int rank);

  bool waitForFutureOrTimeout(
      std::future<bool>& fut,
      const std::chrono::milliseconds& timeOutMilSec,
      const std::string& futDescription,
      ::c10d::C10dLoggingData& debugLog,
      bool throwException = false);

  // Backend-specific abort method
  virtual bool abortComms(const std::optional<std::string>& abortReason = std::nullopt) = 0;
  void abort() override;

  ErrorType getError() override;
  virtual std::string dump_backend_trace(
      bool includeCollectives,
      bool includeStackTraces,
      bool onlyActive) = 0;


 protected:
  static const int64_t kWatchdogThreadSleepMillis;
  c10::intrusive_ptr<Store> store_;
  c10::intrusive_ptr<Store> globalStore_;
  static std::atomic<bool> shouldDump_;
  size_t local_id_;
  // Whether or not we should terminate the watchdog and workCleanup threads.
  std::atomic<bool> terminateProcessGroup_{false};
  ErrorType error_ = ErrorType::SUCCESS;
  std::mutex errorMutex_;
  std::atomic<bool> enableTiming_;

  uint64_t seqP2P_{0};
  uint64_t op_id_{0};

  // Additional member variables used in implementation
  std::string logPrefix_;
  int traceBufferSize_;
  std::shared_ptr<ProcessGroupStatus> pgStatus_ = std::make_shared<ProcessGroupStatus>();

  std::mutex workListMutex_;

  std::unique_ptr<HeartbeatMonitor> heartbeatMonitor_;
  std::unique_ptr<Watchdog> watchdog_;
};

template <typename EventType>
std::string ProcessGroupInterface::WorkInterface::getTraceback() const {
  // First step we get the corresponding record entry from FR, based on work's
  // trace_id_
  std::optional<typename FlightRecorder<EventType>::Entry> entry =
      FlightRecorder<EventType>::get()->getEntry(trace_id_, trace_reset_epoch_);
  if (entry.has_value()) {
    auto entryVal = entry.value();
    // Get stack trace from FR entry, in string format
    // Note: `getTraceback` call below invokes `torch::symbolize`, which may
    // need to acquire the GIL. In order for watchdog to be block-free, we make
    // the call with std::async.
    auto future = std::async(
        std::launch::async, [&entryVal]() { return entryVal.getTraceback(); });
    // Wait for the future to complete or timeout
    auto status = future.wait_for(std::chrono::seconds(8));
    if (status == std::future_status::ready) {
      return future.get();
    }
  }
  return "";
}

// Print the traceback of the collective at call time
template <typename EventType>
void ProcessGroupInterface::WorkInterface::printTraceback() const {
  std::string tracebackStr = getTraceback<EventType>();
  if (!tracebackStr.empty()) {
    LOG(ERROR) << "Stack trace of the failed collective: \n" << tracebackStr;
  } // else, symbolizer probably timed out, we skip logging the stack trace.
  else {
    LOG(ERROR)
        << "Stack trace of the failed collective not found, "
        << "potentially because FlightRecorder is disabled. "
        << "You can enable it by setting TORCH_NCCL_TRACE_BUFFER_SIZE to a non-zero value.";
  }
}

TORCH_API std::optional<
    std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper();
typedef bool (*gil_checker_t)();
TORCH_API gil_checker_t& get_gil_checker();

} // namespace c10d
#endif // USE_C10D_XCCL
