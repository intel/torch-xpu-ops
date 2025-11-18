#ifdef USE_C10D_XCCL

#include <c10/util/thread_name.h>
#include <c10/util/WaitCounter.h>
#include <xccl/ProcessGroupXCCL.hpp>
namespace c10d {

using FlightRecorderXCCL = FlightRecorder<at::xpu::XPUEvent>;

static long computeDeltaMS(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

const int64_t ProcessGroupInterface::kWatchdogThreadSleepMillis = 100;
std::atomic<bool> ProcessGroupInterface::shouldDump_(false);

std::optional<std::function<void(std::function<void(const std::string&)>)>>&
get_cpp_trace_dumper() {
  static std::optional<
      std::function<void(std::function<void(const std::string&)>)>>
      dumper(std::nullopt);
  return dumper;
}

gil_checker_t& get_gil_checker() {
  static gil_checker_t gil_checker = nullptr;
  return gil_checker;
}

static std::future<bool> launchAsyncGilCheck() {
  std::promise<bool> resultPromise;
  std::future<bool> resultFuture = resultPromise.get_future();
  TORCH_CHECK(get_gil_checker(), "Can't check GIL with null GIL checker");
  std::thread workerThread([promise = std::move(resultPromise)]() mutable {
    c10::setThreadName("pt_nccl_gil_chk");

    try {
      auto& gil_checker = get_gil_checker();
      promise.set_value((*gil_checker)());
    } catch (...) {
      promise.set_exception(std::current_exception());
    }
  });

  // Detach the thread to allow it to run independently
  workerThread.detach();

  return resultFuture;
}

std::ostream& operator<<(
    std::ostream& output,
    const ProcessGroupInterface::WorkInterface& work) {
  std::string workInfo;
  workInfo = c10::str(
      "Work(",
      "SeqNum=",
      work.seq_,
      ", OpType=",
      opTypeToString(work.opType_),
      ", NumelIn=",
      work.numelIn_,
      ", NumelOut=",
      work.numelOut_,
      ", Timeout(ms)=",
      work.opTimeout_.count(),
      ")");
  return output << workInfo;
}

static std::atomic<size_t> process_group_id = 0;

ProcessGroupInterface::ProcessGroupInterface(int rank, int size, c10::intrusive_ptr<Store> store)
    : Backend(rank, size),
      local_id_(process_group_id++),
      store_(std::move(store)) {}

const std::string& ProcessGroupInterface::logPrefix() const {
  return logPrefix_;
}

const int& ProcessGroupInterface::globalRank() const {
  static int globalRank = rank_;
  return globalRank;
}

uint64_t ProcessGroupInterface::getUid() {
  return static_cast<uint64_t>(local_id_);
}

uint64_t ProcessGroupInterface::getWatchdogHeartbeat() const {
  return watchdog_->getHeartbeat();
}

ErrorType ProcessGroupInterface::getError() {
  std::lock_guard<std::mutex> lock(errorMutex_);
  return error_;
}

void ProcessGroupInterface::terminateProcess(const std::string& errMsg) {
  // Logging with `FATAL`, after errMsg printed, it calls `std::abort()`
  // to terminate the program execution.
  LOG(FATAL) << logPrefix() << errMsg;
}

bool ProcessGroupInterface::dumpDebuggingInfo(
    bool includeStackTrace /*=true*/,
    bool onlyActive /*=false*/) {
  // This will log counter for how long dumpDebuggingInfo actually takes.
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupInterface__dumpDebuggingInfo);

  // Serialize all calls to this function to avoid corrupting data, but allow
  // multiple calls in one runtime. User is responsible for preserving the
  // output file from an earlier call before a later call overwrites it.
  static std::mutex writeDebugInfoMutex;
  LOG(ERROR)
      << logPrefix()
      << "ProcessGroup" << getBackendName()
      << " preparing to dump debug info. Include stack trace: "
      << includeStackTrace << ", only active collectives: " << onlyActive;
  if (traceBufferSize_ > 0) {
    // We dump nccl trace into local disk by default and users can register
    // their customized writer by inheriting `DebugInfoWriter` via
    // `registerDebugInfoWriter`.
    auto backendTrace = dump_backend_trace(true, includeStackTrace, onlyActive);
    // dump_backend_trace will hang so we don't grab the global lock until we get
    // the trace.
    std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(globalRank());
    LOG(INFO) << logPrefix() << "ProcessGroup" << getBackendName()
              << " dumping backend trace to "
              << writer.getWriterTarget();
    writer.write(backendTrace);
    LOG(INFO) << logPrefix() << "Flight Recorder trace successfully dumped.";
    return true;
  }
  return false;
}

void ProcessGroupInterface::dumpExtraDebuggingInfo() {
  // This extra dump is intended to capture the current snapshot of collectives
  // When this process group is terminated for some exception out of backend CCL
  bool dumpExtraOnExec_ = getCvarBool(EXTRA_DUMP_ON_EXEC, false);
  if (dumpExtraOnExec_) {
    bool should_dump_local = false;
    bool succeeded = shouldDump_.compare_exchange_strong(
        should_dump_local,
        true,
        std::memory_order_release,
        std::memory_order_acquire);
    if (succeeded) {
      LOG(INFO) << logPrefix() << "Sending extra dumping signal";
      broadcastDumpSignal();
      // When this routine is called, exception is captured so
      // dumping by default_pg is not guaranteed due to early termination of
      // process So we call dumping manually here
      bool onlyActive = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
      // Stacktrace is not included at the moment to prevent deadlock due to GIL
      dumpDebuggingInfo(false, onlyActive);
    }
  }
}

void ProcessGroupInterface::broadcastDumpSignal() {
  // broadcast dump signal to all other global ranks.
  broadcastSignal(globalStore_, std::string(kStoreDumpKey), globalRank());
  // signal the local rank to start dumping
  if (!shouldDump_.load()) {
    LOG(ERROR) << logPrefix() << "First PG on this rank to signal dumping.";
    // signal the monitor thread on PG0 to start dumping
    shouldDump_.store(true);
  }
}

void ProcessGroupInterface::broadcastSignal(
    c10::intrusive_ptr<Store>& store,
    const std::string& signal,
    int srcRank) {
  try {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(&srcRank),
        reinterpret_cast<uint8_t*>(&srcRank) + sizeof(srcRank));
    store->set(signal, vec);
    LOG(INFO) << logPrefix() << "Broadcasting signal " << signal
              << " to other ranks via TCPStore.";
  } catch (const std::exception& e) {
    LOG(ERROR) << logPrefix() << "Failed to broadcast signal " << signal
               << " through TCPStore. Error: " << e.what();
  }
}

bool ProcessGroupInterface::waitForFutureOrTimeout(
    std::future<bool>& fut,
    const std::chrono::milliseconds& timeOutMilSec,
    const std::string& futDescription,
    ::c10d::C10dLoggingData& debugLog,
    bool throwException) {
  std::string errorMsg;
  bool complete = false;

  TORCH_CHECK(fut.valid(), "Expected a valid future");
  std::future_status status = fut.wait_for(timeOutMilSec);
  if (status == std::future_status::ready) {
    // Calling .get() will re-raise any exception from the future, and we don't
    // care about the retval
    try {
      bool result = fut.get();
      if (result) {
        VLOG(2) << logPrefix()
                << "future successfully executed for: " << futDescription;
        debugLog.strings["status"] = "SUCCESS";
        complete = true;
      }
    } catch (const std::exception& e) {
      errorMsg = c10::str(
          logPrefix(),
          "Exception thrown when waiting for future ",
          futDescription,
          ": ",
          e.what());

      debugLog.strings["status"] = "EXCEPTION";
      debugLog.strings["exception_msg"] = e.what();
      LOG(ERROR) << errorMsg;
    } catch (...) {
      errorMsg = c10::str(
          logPrefix(),
          "Unknown exception thrown when waiting for future ",
          futDescription);
      debugLog.strings["status"] = "EXCEPTION";
      debugLog.strings["exception_msg"] = "Unknown exception";
      LOG(ERROR) << errorMsg;
    }
  } else {
    errorMsg = c10::str(
        logPrefix(),
        "Future for ",
        futDescription,
        " timed out after ",
        timeOutMilSec.count(),
        " ms");
    debugLog.strings["status"] = "TIMEOUT";
    LOG(ERROR) << errorMsg;
  }
  if (throwException && !errorMsg.empty()) {
    C10_THROW_ERROR(DistBackendError, errorMsg);
  }
  return complete;
}

// Abort this backend.
void ProcessGroupInterface::abort() {
  // This will log counter for how long the abort actually takes.
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupInterface__abort);

  dumpExtraDebuggingInfo();
  // Don't join threads here since the purpose of this method is to abort all
  // communicators and signal the threads to exit. Joining on the threads could
  // potentially block and hence avoid it in this method.
  terminateProcessGroup_.store(true);
  watchdog_->notify();
  // launch abort asynchronously and wait for it to complete or timeout
  LOG(INFO) << logPrefix()
            << "Launching ProcessGroup" << getBackendName()
            << " abort asynchronously.";
  std::future<bool> fut =
      std::async(std::launch::async, [this]() { return this->abortComms(); });

  ::c10d::C10dLoggingData debugLog;
  waitForFutureOrTimeout(
      fut, getOptionsTimeout(), "ProcessGroup abort", debugLog, true);
  LOG(INFO) << logPrefix() << "ProcessGroup" << getBackendName() << " aborts successfully.";

  // We need to wait for abort to finish before we can safely shut down
  // heartbeat monitoring thread.
  heartbeatMonitor_->stop();
}


ProcessGroupInterface::WorkInterface::WorkInterface(
  int rank,
  OpType opType,
  uint64_t seq,
  const char* profilingTitle,
  const std::optional<std::vector<at::Tensor>>& inputs,
  bool enableTiming)
    : Work(rank, opType, profilingTitle, inputs),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq),
      timingEnabled_(enableTiming) {}

ProcessGroupInterface::WorkInterface::WorkInterface(const WorkInterface& w)
    : Work(w.rank_, w.opType_),
      opTimeout_(w.opTimeout_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      startTraceUpdated_(w.startTraceUpdated_),
      numelIn_(w.numelIn_),
      numelOut_(w.numelOut_),
      futureWorkResult_(w.futureWorkResult_),
      timingEnabled_(w.timingEnabled_),
      trace_id_(w.trace_id_),
      trace_reset_epoch_(w.trace_reset_epoch_) {
  exception_ = w.exception_;
}

void ProcessGroupInterface::WorkInterface::setException(
    std::exception_ptr exception_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);
  exception_ = std::move(exception_ptr);
}

bool ProcessGroupInterface::WorkInterface::checkTimeout(
    std::optional<std::chrono::milliseconds> timeout) {
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.wait_counter.ProcessGroupWorkInterface__checkTimeout);
  auto currentTimepoint = std::chrono::steady_clock::now();
  auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTimepoint - workStartTime_);
  auto workTimeout = timeout ? *timeout : opTimeout_;

  if (timeElapsed < workTimeout) {
    return false;
  }

  // Timed out

  std::string exceptionMsg = c10::str(
      logPrefix(),
      "Watchdog caught collective operation timeout: ",
      *this,
      " ran for ",
      timeElapsed.count(),
      " milliseconds before timing out.");

  LOG(ERROR) << exceptionMsg;

  std::exception_ptr exception_ptr =
      std::make_exception_ptr(C10_BUILD_ERROR(DistBackendError, exceptionMsg));
  if (!exception()) {
    // if there is already an error, we don't override it
    setException(exception_ptr);
  }

  // Mark future result as TIMEOUT
  if (futureWorkResult_ && !futureWorkResult_->completed()) {
    futureWorkResult_->markCompleted(
        at::IValue(static_cast<uint8_t>(WorkResult::TIMEOUT)));
  }
  return true;
}

ProcessGroupInterface::HeartbeatMonitor::HeartbeatMonitor(ProcessGroupInterface* pg) {
  pg_ = pg;
  heartbeatTimeoutInSec_ = getCvarInt(HEARTBEAT_TIMEOUT_SEC, 60 * 8 /*8 Mins*/);
  waitTimeoutDumpInMilSec_ = getCvarInt(WAIT_TIMEOUT_DUMP_MILSEC, 15 * 1000 /*15 Sec*/);
  coordCheckIntervalMilSec_ = getCvarInt(COORD_CHECK_MILSEC, 1000);
  dumpOnTimeoutOrEx_ = getCvarBool(DUMP_ON_TIMEOUT, true);
  logCppStackOnUncleanShutdown_ = getCvarBool(LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN, true);
  watchdogHeartbeatMonitorEnabled_ = getCvarBool(ENABLE_MONITORING, true);

  // Print out ENV settings for the heartbeat monitor thread
  LOG(INFO)
      << pg_->logPrefix() << "HeartbeatMonitor environments: "
      << "TORCH_PG_ENABLE_MONITORING (Whether to kill program when no watchdog heartbeat detected): "
      << watchdogHeartbeatMonitorEnabled_
      << ", TORCH_PG_DUMP_ON_TIMEOUT: " << dumpOnTimeoutOrEx_
      << ", TORCH_PG_WAIT_TIMEOUT_DUMP_MILSEC: " << waitTimeoutDumpInMilSec_
      << ", TORCH_PG_HEARTBEAT_TIMEOUT_SEC: " << heartbeatTimeoutInSec_
      << ", TORCH_PG_COORD_CHECK_MILSEC: " << coordCheckIntervalMilSec_
      << ", TORCH_PG_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN: "
      << logCppStackOnUncleanShutdown_;
}

std::string ProcessGroupInterface::HeartbeatMonitor::getWatchdogTimeoutErrorMsg(const std::string& extraMsg) {
  return c10::str(
      pg_->logPrefix(),
      "Received a dump signal due to a collective timeout from ",
      extraMsg,
      " and we will try our best to dump the debug info. ",
      "Last enqueued ", pg_->getBackendName(), " work: ",
      pg_->pgStatus_->lastEnqueuedSeq,
      ", last completed ", pg_->getBackendName(), " work: ",
      pg_->pgStatus_->lastCompletedSeq,
      ".",
      "This is most likely caused by incorrect usages of collectives, e.g., wrong ",
      "sizes used across ranks, the order of collectives is not same for all ranks ",
      "or the scheduled collective, for some reason, didn't run. Additionally, ",
      "this can be caused by GIL deadlock or other reasons such as network errors or ",
      "bugs in the communications library (e.g. NCCL), etc. ");
}
std::string ProcessGroupInterface::HeartbeatMonitor::getWatchdogTimeoutExitMsg(const std::string& exitReason) {
  return c10::str(
      pg_->logPrefix(),
      "Terminating the process after attempting to dump debug info, due to ",
      exitReason,
      ".");
}

int ProcessGroupInterface::HeartbeatMonitor::getDumpTimeout() const {
  return waitTimeoutDumpInMilSec_;
}

void ProcessGroupInterface::HeartbeatMonitor::stop() {
  terminateHeartbeatMonitorThread_.store(true);
  monitorWakeUpCV_.notify_one();
}

void ProcessGroupInterface::HeartbeatMonitor::start() {
  TORCH_CHECK(
      !heartbeatMonitorThread_.joinable(),
      "HeartbeatMonitor thread already started");
  heartbeatMonitorThread_ =
      std::thread(&ProcessGroupInterface::HeartbeatMonitor::runLoop, this);
}

void ProcessGroupInterface::HeartbeatMonitor::join() {
  if (heartbeatMonitorThread_.joinable()) {
    heartbeatMonitorThread_.join();
    LOG(INFO) << pg_->logPrefix()
              << "Process Group heart beat monitor thread joined.";
  }
}

void ProcessGroupInterface::HeartbeatMonitor::runLoop() {
  c10::setThreadName("pt_pg_heartbt");
  STATIC_SCOPED_WAIT_COUNTER(
      pytorch.ProcessGroupInterface__HeartbeatMonitor__runLoop);

  uint64_t heartBeatCounter = 0ULL;
  std::string errorMsg;
  std::string exitReason;
  bool checkDumpSignal = (dumpOnTimeoutOrEx_ && pg_->getUid() == 0);
  int monitorPollInterval = checkDumpSignal ? coordCheckIntervalMilSec_
                                            : heartbeatTimeoutInSec_ * 1000;
  auto lastTimePollStore = std::chrono::steady_clock::now();
  auto lastTimeHeartBeatCheck = std::chrono::steady_clock::now();
  std::optional<DumpPipe> dumpPipe = std::nullopt;
  // Use a pool to temporarily store the futures to avoid blocking when the code
  // exits the scope of when future is generated by std::async.
  std::vector<std::future<bool>> futures;

  if (pg_->getUid() == 0) {
    // DumpPipe is one per-trainer process, and its convenient to name them
    // after 'global' ranks in the system, So we assume processgroup (uid)==0 is
    // the global PG and has globally unique rank ids across trainers.
    dumpPipe.emplace(pg_->globalRank());
  }

  while (true) {
    // This won't have any lock since this lock is only used here.
    // Please be aware that mutex `monitorMutex_` should not be used
    // somewhere else to avoid the deadlock.
    std::unique_lock<std::mutex> lock(monitorMutex_);
    if (monitorWakeUpCV_.wait_for(
            lock, std::chrono::milliseconds(monitorPollInterval), [&] {
              return terminateHeartbeatMonitorThread_.load();
            })) {
      // For the normal complete or user interception, monitorWakeUpCV_
      // will get notified, we early return and exit heartbeatMonitor.
      return;
    }
    auto currentTime = std::chrono::steady_clock::now();

    // We put extra functionality in the thread for the default PG (aka,
    // local_id_=0) because the signal is same across different PGs. We only
    // need to run once per process to avoid duplicate things performed in too
    // many separate threads. For example, we check a global flag on the
    // TCPStore periodically to see if any PG on any rank observed a timeout and
    // signaled peers to dump debugging info, and we avoid hammering the
    // TCPStore from all PGs on the same rank.
    if (checkDumpSignal) {
      // There are two scenarios where monitor thread will dump on timeout:
      // 1. The current rank is the first to observe a timeout in watchdog.
      // (shouldDump_ was set to true by the watchdog thread).
      // 2. Other ranks detected the timeout and signal the current rank to
      // dump. In addition, monitor threads will dump if watchdog threads has no
      // heartbeat or dumpPipe is not empty.
      if (pg_->shouldDump_.load()) {
        errorMsg = getWatchdogTimeoutErrorMsg("this local rank");
        exitReason = "collective timeout or exception";
        break;
      }
      // We poll store to see if some ranks have flagged a timeout when
      // we haven't polled for `heartbeat_timeout` seconds and there haven't
      // any work added or removed for `watchdog_timeout` seconds.
      if (computeDeltaMS(lastWorkListUpdateTime_, currentTime) >=
              ProcessGroupInterface::kWatchdogThreadSleepMillis &&
          computeDeltaMS(lastTimePollStore, currentTime) >=
              coordCheckIntervalMilSec_) {
        lastTimePollStore = currentTime;
        auto handleError = [&](const std::string& errorMessage) {
          LOG(WARNING)
              << pg_->logPrefix()
              << "Failed to check the \"should dump\" flag on Store, "
              << "(maybe Store server has shut down too early), with error: "
              << errorMessage;
          // We give up for now assuming TCPStore has been torn down.
          return;
        };
        // Wrap globalStore_->check() in a try-catch block to avoid crashing if
        // the store is not available.
        bool checkExceptionDump = false;
        try {
          checkExceptionDump = pg_->globalStore()->check({std::string(kStoreDumpKey)});
        } catch (const c10::DistNetworkError& e) {
          handleError(e.msg());
        } catch (const std::exception& e) {
          handleError(e.what());
        }

        if (checkExceptionDump) {
          int timeOutRank = -1;
          if (!pg_->shouldDump_.load()) {
            LOG(ERROR)
                << pg_->logPrefix()
                << "Observed flight recorder dump signal from another rank via TCPStore.";
          }
          pg_->shouldDump_.store(true);
          try {
            auto vec = pg_->globalStore()->get(std::string(kStoreDumpKey));
            TORCH_CHECK_WITH(
                DistBackendError,
                vec.size() == sizeof(int),
                "Invalid size for the timeout rank ID");
            std::memcpy(&timeOutRank, vec.data(), vec.size());
          } catch (const std::exception& e) {
            LOG(ERROR) << pg_->logPrefix()
                       << "Failed to get timeout rank ID from TCPStore."
                       << e.what();
          }
          errorMsg = getWatchdogTimeoutErrorMsg(c10::str(" rank ", timeOutRank));
          exitReason = "collective timeout or exception";
          break;
        }
      }
    }

    if (computeDeltaMS(lastTimeHeartBeatCheck, currentTime) >=
        heartbeatTimeoutInSec_ * 1000l) {
      // Check the heart beat of watchdog thread.
      lastTimeHeartBeatCheck = currentTime;
      auto heartbeat = pg_->getWatchdogHeartbeat();
      if (heartbeat != heartBeatCounter) {
        heartBeatCounter = heartbeat;
      } else {
        pg_->shouldDump_.store(true);
        // Watchdog heartbeat timeout
        errorMsg = c10::str(
            pg_->logPrefix(),
            "ProcessGroup", pg_->getBackendName(), "'s watchdog got stuck for ",
            heartbeatTimeoutInSec_,
            " seconds without making progress in monitoring enqueued collectives. ",
            "This typically indicates a ", pg_->getBackendName(), " API hang blocking the watchdog, ",
            "and could be triggered by another thread holding the GIL inside a ",
            "device api, or other deadlock-prone behaviors.",
            "If you suspect the watchdog is not actually stuck and a longer timeout would help, ",
            "you can either increase the timeout (TORCH_PG_HEARTBEAT_TIMEOUT_SEC) to a larger value "
            "or disable the heartbeat monitor (TORCH_PG_ENABLE_MONITORING=0)."
            "If either of aforementioned helps, feel free to file an issue to PyTorch about the short timeout "
            "or false positive abort; otherwise, please attempt to debug the hang. ");
        exitReason = c10::str("ProcessGroup", pg_->getBackendName(), " watchdog hang");
        break;
      }
    }
    // process a request to dump the trace. only PG uid 0 will respond to dump
    // requests, but this is fine since all PG's feed into the same flight
    // recorder and dump. After dump, the training should continue.
    if (dumpPipe.has_value() && dumpPipe->shouldDump()) {
      // best effort dump, not waiting for the dump here
      bool onlyActive = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
      LOG(INFO) << pg_->logPrefix()
                << "Dump signal received through pipe, triggering FR dump.";
      futures.emplace_back(std::async(std::launch::async, [this, onlyActive]() {
        return this->pg_->dumpDebuggingInfo(false, onlyActive);
      }));
    }
  }
  LOG(ERROR) << errorMsg;

  // We perform some checks to help users debug the timeout/hang issue:
  // 1. Dump the flight recorder trace to help debug the issue
  //    (timeout after waitTimeoutDumpInMilSec_, which is one minute).
  // 2. Check if there is a GIL deadlock (timeout after 300ms).
  // 3. Try to dump the c++ stacktraces (blocking and would hang,
  //    users can turn this off by set
  //    TORCH_FR_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN=0).

  // Dump debugging info if needed
  if (checkDumpSignal && pg_->shouldDump_.load()) {
    // Store debug info to storage if no other thread does it. (By default to
    // local disk)
    bool dumpStackTrace = getCvarBool(TORCH_INCLUDE_STACK_TRACE, true);
    bool onlyActive = getCvarBool(TORCH_INCLUDE_ONLY_ACTIVE, false);
    ::c10d::C10dLoggingData debugLog;
    debugLog.integers["pg_id"] = static_cast<int64_t>(pg_->getUid());
    debugLog.integers["rank"] = pg_->getRank();
    debugLog.integers["global_rank"] = pg_->globalRank();
    debugLog.integers["world_size"] = pg_->getSize();
    debugLog.strings["flight_recorder_version"] = c10d::version_val_str;
    for (int i = 0; i < 2; i++) {
      std::future<bool> asyncDebugDump =
          std::async(std::launch::async, [this, dumpStackTrace, onlyActive]() {
            return this->pg_->dumpDebuggingInfo(dumpStackTrace, onlyActive);
          });

      // wait for the dump until timeout - log data
      auto complete = pg_->waitForFutureOrTimeout(
          asyncDebugDump,
          std::chrono::milliseconds(waitTimeoutDumpInMilSec_),
          "Flight recorder dump in heartbeatMonitor",
          debugLog,
          false);

      if (complete) {
        LOG(INFO)
            << pg_->logPrefix()
            << "Finished flight recorder successfully. Output can be analyzed using the fr_trace script.";
        if (i > 0) {
          debugLog.strings["exception_msg"] = "Dump with stack trace failed.";
        }
        break;
      }
      // If we failed to dump, try dumping without stack trace in the 2nd
      // iteration.
      dumpStackTrace = false;
      futures.emplace_back(std::move(asyncDebugDump));
    }
    debugLog.integers["trace_enabled"] = int64_t(dumpStackTrace);
    auto logger = c10d::C10dLogger::getLogger();
    if (logger) {
      logger->log(debugLog);
    }
  }

  // GIL deadlock check.
  if (get_gil_checker() != nullptr) {
    auto fut = launchAsyncGilCheck();
    auto kGilCheckTimeout = std::chrono::milliseconds(300);
    auto futStatus = fut.wait_for(kGilCheckTimeout);
    if (futStatus != std::future_status::ready) {
      TORCH_CHECK(
          futStatus != std::future_status::deferred,
          "Expected the future to have been launched eagerly.");
      LOG(ERROR)
          << pg_->logPrefix()
          << "Could not acquire GIL within 300 ms on exit, possible GIL induced hang";
    }
  } else {
    VLOG(2)
        << pg_->logPrefix()
        << "GIL checker was not registered, perhaps this is a no-python build?";
  }

  // Dump the c++ stacktraces.
  auto& cpp_dumper = get_cpp_trace_dumper();
  if (logCppStackOnUncleanShutdown_ && cpp_dumper.has_value()) {
    LOG(INFO) << pg_->logPrefix() << "Dumping c++ stacktraces:";
    cpp_dumper.value()([&](const std::string& line) {
      LOG(INFO) << pg_->logPrefix() << line;
    });
    LOG(INFO) << pg_->logPrefix() << "Finished c++ stacktraces dump.";
  }

  // There are two possible cases for the watchdog thread exit:
  // Case one: desync report runs quickly, and it follows the step:
  // collective timeout -> desync -> exception handling -> throwing exception.
  // The program will exit because of exception thrown and the code below will
  // not be run.
  //
  // Case two: desync might be slow or get stuck and we need to wait
  // extra time to avoid we kill the program too early.
  //
  // Or we get stuck in destructors, we will sleep for some time before calling
  // std::abort() to kill the whole process.
  if ((pg_->terminateProcessGroup_.load() || pg_->shouldDump_.load()) &&
      !terminateHeartbeatMonitorThread_.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(heartbeatTimeoutInSec_));
    LOG(INFO)
        << pg_->logPrefix() << "slept for " << heartbeatTimeoutInSec_
        << " because we want to wait longer to verify there is indeed a watchdog hang.";
  }

  // At this point, we either already sleep for another `heartbeatTimeoutInSec_`
  // or the thread has finished. Because we don't want to block the monitor
  // thread, so We mark the thread detach and the dump of debug info becomes
  // "best effort". If the process exit normally, marking it detach also makes
  // sense because we don't really care about dumping the debug info.

  // We already log completion inside the thread, so it may not be necessary to
  // check the return value here.  We mainly use a future so we can exit early
  // if done.
  if (!terminateHeartbeatMonitorThread_.load()) {
    // Create a error message reported from MonitorThread, so
    // we throw exception and make the whole process to be killed.
    // TODO(fduwjj): After having a hang debug wiki, we need to update the wiki
    // url here.
    if (watchdogHeartbeatMonitorEnabled_) {
      pg_->terminateProcess(getWatchdogTimeoutExitMsg(exitReason));
    } else {
      // Ideally we want to merge this one with the above one, but we are going
      // to remove the kill switch for monitor thread soon, so we keep this one
      // for now.
      LOG(ERROR)
          << pg_->logPrefix()
          << "ProcessGroup monitor thread is disabled, but would have terminated the process"
          << "after attempting to dump debug info, due to " << exitReason << ".";
    }
  }
}

// Watchdog implementation
ProcessGroupInterface::Watchdog::Watchdog(ProcessGroupInterface* pg) : pg_(pg) {
  heartbeat_ = 1ULL;
  rethrowBackendErrors_ = getCvarBool(RETHROW_BACKEND_ERRORS, true);
  propagatePgError_ = getCvarBool(PROPAGATE_ERROR, false);
  desyncDebug_ = getCvarBool(DESYNC_DEBUG, false) ||
      (pg_->dist_debug_level_ >= DebugLevel::Detail);

  // print out ENV settings for the watchdog thread.
  LOG(INFO) << pg_->logPrefix() << "ProcessGroup Watchdog environments: "
            << "TORCH_PG_RETHROW_BACKEND_ERRORS: " << rethrowBackendErrors_
            << ", TORCH_PG_PROPAGATE_ERROR: " << propagatePgError_
            << ", TORCH_PG_DESYNC_DEBUG: " << desyncDebug_;

  // Enable Desync Debugger per user setting
  if (desyncDebug_) {
    desyncDebugger_.init(pg_);
  }
}

void ProcessGroupInterface::Watchdog::notify() {
  watchdogCV_.notify_one();
}

void ProcessGroupInterface::Watchdog::start() {
  TORCH_CHECK(!watchdogThread_.joinable(), "Watchdog thread already started");
  watchdogThread_ = std::thread(&ProcessGroupInterface::Watchdog::run, this);
}

void ProcessGroupInterface::Watchdog::join() {
  if (watchdogThread_.joinable()) {
    watchdogThread_.join();
    LOG(INFO) << pg_->logPrefix() << "ProcessGroup"
              << pg_->getBackendName() << " watchdog thread joined.";
  }
}

void ProcessGroupInterface::Watchdog::run() {
  c10::setThreadName("pt_pg_watchdg");
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupInterface__Watchdog__run)

  try {
    VLOG(2) << pg_->logPrefix() << "Process group watchdog thread started!";
    // Start heartbeatMonitor thread in ProcessGroup, not here
    runLoop();
    VLOG(2) << pg_->logPrefix()
            << "Process group watchdog thread terminated normally";
  } catch (std::exception& e) {
    // This condition is triggered when any routine in watchdog gets an exception
    pg_->dumpExtraDebuggingInfo();

    if (std::string(e.what()).find("driver shutting down") != std::string::npos) {
      VLOG(2)
          << pg_->logPrefix()
          << "main process destroyed device before watchdog loop exited, terminating watchdog."
          << " (Watchdog caught exception: " << e.what();
    } else {
      // Append error message reported from runLoop
      const auto exitMsg = c10::str(
          pg_->logPrefix(),
          "Process group watchdog thread terminated with exception: ",
          e.what());
      LOG(ERROR) << exitMsg;

      if (C10_LIKELY(rethrowBackendErrors_) ||
          !(std::string(e.what()).find(backendErrorString_))) {
        // TODO(whc) clean up the rethrow - why is it stored in a class var and
        // rethrown?
        watchDogException_ = std::make_exception_ptr(
            C10_BUILD_ERROR(DistBackendError, exitMsg));
        std::rethrow_exception(watchDogException_);
      }
    }
  } catch (...) {
    const auto exitMsg = c10::str(
        pg_->logPrefix(),
        "Process group watchdog thread terminated with exception: unknown");
    LOG(ERROR) << exitMsg;
    watchDogException_ = std::make_exception_ptr(
        C10_BUILD_ERROR(DistBackendError, exitMsg));
    std::rethrow_exception(watchDogException_);
  }
}

int ProcessGroupInterface::Watchdog::getSignalSrcRank(
    c10::intrusive_ptr<Store>& store,
    const std::string& signal) {
  // This function is 'non blocking'. We first 'check' if the key exists in the
  // store, then read/get the value only if the key exists.
  int srcRank = -1;
  bool signalExists = false;
  try {
    signalExists = store->check({signal});
  } catch (const std::exception& e) {
    LOG(WARNING) << pg_->logPrefix() << "Failed to check the signal " << signal
                 << " on TCPStore, " << e.what();
  }
  if (!signalExists) {
    return srcRank;
  }

  // Key exists, now read and parse the value (source rank)
  std::vector<uint8_t> vec;
  try {
    vec = store->get(std::string(signal));
  } catch (const std::exception& e) {
    LOG(ERROR) << pg_->logPrefix() << "Failed to get source rank of the signal "
               << signal << " from TCPStore." << e.what();
  }
  TORCH_CHECK_WITH(
      DistBackendError,
      vec.size() == sizeof(int),
      "Invalid size for the timeout rank ID");
  std::memcpy(&srcRank, vec.data(), vec.size());
  return srcRank;
}

void ProcessGroupInterface::Watchdog::checkAndSetRemoteError() {
  // if the error is already set, no need to check again
  if (pg_->getError() != ErrorType::SUCCESS) {
    return;
  }
  // key/signal to read from the store is a string and pg specific:
  // format is: remote_error:pg_uid
  int remoteErrorRank = getSignalSrcRank(
      pg_->store_,
      std::string(kStoreErrorSignalKey) + ':' + pg_->getGroupUid());
  if (remoteErrorRank != -1) {
    std::lock_guard<std::mutex> lock(pg_->errorMutex_);
    pg_->error_ = ErrorType::REMOTE_ERROR;
    LOG(ERROR) << c10::str(
        pg_->logPrefix(),
        " remote error detected from rank: ",
        remoteErrorRank);
  }
}

uint64_t ProcessGroupInterface::Watchdog::getHeartbeat() const {
  return heartbeat_.load();
}

void ProcessGroupInterface::Watchdog::setDesyncDebug(bool desyncDebug) {
  desyncDebug_ = desyncDebug;
}

void ProcessGroupXCCL::WatchdogXCCL::runLoop() {
  bool done = false;
  pg_->heartbeatMonitor_->setLastWorkListUpdateTime();
  auto lastStatusUpdateTime = std::chrono::steady_clock::now();

  while (!done || !pg_->terminateProcessGroup_.load()) {
    std::unique_lock<std::mutex> lock(pg_->workListMutex_);
    // We busy-poll the work vector every kWatchdogThreadSleepMillis
    // milliseconds as long as the atomic is True.
    watchdogCV_.wait_for(
        lock,
        std::chrono::milliseconds(kWatchdogThreadSleepMillis),
        [&]() -> bool { return pg_->terminateProcessGroup_.load(); });
    // Bump up heart beat by one.
    heartbeat_++;

    // Log the progress of this PG periodically
    auto logger = ::c10d::C10dLogger::getLogger();
    if (logger &&
        computeDeltaMS(lastStatusUpdateTime, std::chrono::steady_clock::now()) >=
            kWorkStatusUpdatePeriodMs) {
      ::c10d::C10dLoggingData data;
      data.integers["pg_id"] = static_cast<int64_t>(pg_->getUid());
      data.integers["rank"] = pg_->getRank();
      data.integers["global_rank"] = pg_->globalRank();
      data.strings["pg_backend"] = pg_->getBackendName();
      // TODO: (frost-intel) Track started status so we can combine this with PGNCCL
      data.integers["last_enqueued_work"] = pg_->pgStatus_->lastEnqueuedSeq;
      data.integers["last_completed_work"] = pg_->pgStatus_->lastCompletedSeq;
      data.integers["last_enqueued_numel_in"] =
          static_cast<int64_t>(pg_->pgStatus_->lastEnqueuedNumelIn);
      data.integers["last_enqueued_numel_out"] =
          static_cast<int64_t>(pg_->pgStatus_->lastEnqueuedNumelOut);
      data.integers["last_completed_numel_in"] =
          static_cast<int64_t>(pg_->pgStatus_->lastCompletedNumelIn);
      data.integers["last_completed_numel_out"] =
          static_cast<int64_t>(pg_->pgStatus_->lastCompletedNumelOut);
      // logging strings
      data.strings["last_enqueued_work_name"] =
          pg_->pgStatus_->lastEnqueuedWorkName;
      data.strings["last_completed_work_name"] =
          pg_->pgStatus_->lastCompletedWorkName;
      logger->log(data);
      lastStatusUpdateTime = std::chrono::steady_clock::now();
    }

    if (propagatePgError_) {
      // Check and set remote error if it has not been set before
      checkAndSetRemoteError();
    }

    // Process work items through the interface
    processWorkList();
    done = !pg_->workMetaList_.empty();
  }
}

void ProcessGroupXCCL::WatchdogXCCL::processWorkList() {
  for (auto it = pg_->workMetaList_.begin(); it != pg_->workMetaList_.end(); /* no increment */) {
    // Access work by reference like ProcessGroupNCCL to avoid access issues
    auto& work = *it;

    // Skip error checking if process group is terminating
    if (!pg_->terminateProcessGroup_.load()) {
      // Check for exceptions in the work item
      // TODO: Does WorkXCCL have any exception generation mechanism?
      if (work.exception()) {
        // set the error to the first error found
        std::lock_guard<std::mutex> lock(pg_->errorMutex_);
        if (pg_->error_ == ErrorType::SUCCESS) {
          pg_->error_ = ErrorType::COMM_ERROR;
        }
      }

      // Check for timeout (will set exception in work if timed out)
      bool timedout = !work.exception() && work.checkTimeout();
      if (timedout) {
        std::lock_guard<std::mutex> lock(pg_->errorMutex_);
        if (pg_->error_ == ErrorType::SUCCESS) {
          pg_->error_ = ErrorType::TIMEOUT;
        }
        desyncDebugger_.run();
      }

      if (work.exception()) {
        LOG(ERROR) << c10::str(
            pg_->logPrefix(),
            " failure detected by watchdog at work sequence id: ",
            work.seq_,
            " PG status: last enqueued work: ",
            pg_->pgStatus_->lastEnqueuedSeq,
            ", last completed work: ",
            pg_->pgStatus_->lastCompletedSeq);

        // Print the traceback of the collective at call time
        work.printTraceback();

        // broadcast remote error signal to all other ranks in this specific PG.
        // key/signal to write in the tcpstore is a string and pg specific:
        // format is: remote_error:pg_uid
        if (propagatePgError_) {
          pg_->broadcastSignal(
            pg_->store_,
            std::string(kStoreErrorSignalKey) + ":" + pg_->getGroupUid(),
            pg_->rank_);
        }

        // try to notify other ranks via global TCPStore to dump the flight
        // recorder when a collective timeout or exception happens. Flight
        // recorder behavior is independent of desync Debug.
        pg_->broadcastDumpSignal();
        // Give time for dumping before throwing exception for all ranks.
        // It is hard to presume or control what the pattern of watchdog might
        // look like, so it is better to let all ranks universally sleep for a
        // short period of time, in this case, 60 seconds, which is also the
        // maximum time we leave for FR dump.
        std::this_thread::sleep_for(std::chrono::milliseconds(
            pg_->heartbeatMonitor_->getDumpTimeout() * 4));

        // We can't handle exceptions since WorkXCCL doesn't support abort()
      }

      desyncDebugger_.logWorkStart(work);

      // allow watchdog to do an event query on a side thread
      c10::OptionalDeviceGuard gpuGuard(work.xcclEndEvent_->device());

      // a work could be started but not completed, so we should not update
      // lastStartedSeq and lastStartedOpName if the work state is checked
      // multiple times after the start
      if (pg_->pgStatus_->lastStartedSeq < static_cast<int64_t>(work.seq_) &&
          work.isStarted()) {
        pg_->pgStatus_->lastStartedSeq = static_cast<int64_t>(work.seq_);
        pg_->pgStatus_->lastStartedWorkName = opTypeToString(work.opType_);
        pg_->pgStatus_->lastStartedNumelIn = work.numelIn_;
        pg_->pgStatus_->lastStartedNumelOut = work.numelOut_;
      }
    }

    // Check if work is completed and clean up
    if (work.isCompleted()) {
      // Update completion statistics (similar to NCCL)
      auto logger = ::c10d::C10dLogger::getLogger();
      if (pg_->enableTiming_ && logger) {
        ::c10d::C10dLoggingData data;
        // logging integers
        data.strings["collective_duration"] =
            std::to_string(work.getDuration());
        data.integers["global_rank"] = pg_->globalRank();
        data.integers["pg_id"] = static_cast<int64_t>(pg_->getUid());
        data.strings["pg_name"] = pg_->pg_uid_;
        data.strings["pg_desc"] = pg_->pg_desc_;
        data.integers["pg_rank"] = pg_->rank_;
        data.integers["world_size"] = pg_->size_;
        data.strings["comm_backend"] = pg_->getBackendName();
        data.strings["comm_backend_version"] = pg_->getBackendCclVersion();
        // TODO: We see errors for this line, revert it for now.
        data.strings["collective_stack"] = "";
        data.strings["collective_name"] = opTypeToString(work.opType_);
        logger->log(data);
      }

      // Work status logging for desync debug
      desyncDebugger_.logWorkEnd(work);

      if (work.futureWorkResult_ && !work.futureWorkResult_->completed()) {
        work.futureWorkResult_->markCompleted(
            at::IValue(static_cast<uint8_t>(WorkResult::SUCCESS)));
      }

      pg_->pgStatus_->lastCompletedSeq = static_cast<int64_t>(work.seq_);
      pg_->pgStatus_->lastCompletedWorkName = opTypeToString(work.opType_);
      pg_->pgStatus_->lastCompletedNumelIn = work.numelIn_;
      pg_->pgStatus_->lastCompletedNumelOut = work.numelOut_;
      FlightRecorderXCCL::get()->retire_id(work.trace_id_, work.trace_reset_epoch_, true);

      // Remove completed work from the list
      it = pg_->workMetaList_.erase(it);

      // Update heartbeat monitor with work list activity
      pg_->heartbeatMonitor_->setLastWorkListUpdateTime();
    } else {
      // Move to next work item if not completed
      ++it;
    }

    // Increment heartbeat after processing each work item
    heartbeat_++;
  }
}

// DesyncDebugger implementation
void ProcessGroupInterface::DesyncDebugger::init(ProcessGroupInterface* pg) {
  rank_ = pg->getRank();
  size_ = pg->getSize();
  globalRank_ = pg->globalRank();
  pgId_ = static_cast<int>(pg->getUid());
  store_ = std::move(pg->store_);
  enabled_ = true;
  pgBackend_ = pg->getBackendName();
  traceKeyStart_ = getTraceStartKey(pgBackend_, rank_);
  traceKeyEnd_ = getTraceEndKey(pgBackend_, rank_);
}

// Run desync debug. This function is called by watchdog at time of timeout.
void ProcessGroupInterface::DesyncDebugger::run() {
  if (!enabled_)
    return;
  auto logPrefix = c10::str("Rank ", rank_);
  ::c10d::C10dLoggingData log;
  log.integers["pg_id"] = pgId_;
  log.integers["rank"] = rank_;
  log.integers["global_rank"] = globalRank_;
  log.integers["world_size"] = size_;
  // Use this to differentiate between flight recorder and desync debug report.
  log.strings["flight_recorder_version"] = "-1";

  try {
    std::string desyncMsg = retrieveDesyncReport(store_, pgBackend_, rank_, size_);
    log.strings["status"] = "SUCCESS";
    LOG(ERROR) << logPrefix << desyncMsg;
  } catch (const std::exception& e) {
    log.strings["status"] = "EXCEPTION";
    log.strings["exception_msg"] = e.what();
    enabled_ = false;
    LOG(ERROR) << logPrefix
               << " Failed to retrieve TORCH_PG_DESYNC_DEBUG report. "
               << " Please file an issue. Error: " << e.what();
  } catch (...) {
    enabled_ = false;
    log.strings["status"] = "EXCEPTION";
    log.strings["exception_msg"] = "Unknown exception";
    LOG(ERROR)
        << logPrefix
        << " Failed to rerieve TORCH_PG_DESYNC_DEBUG report with unknown error."
        << " Please file an issue.";
  }
  auto logger = c10d::C10dLogger::getLogger();
  if (logger) {
    logger->log(log);
  }
}

// Log work start to store.
void ProcessGroupInterface::DesyncDebugger::logWorkStart(ProcessGroupInterface::WorkInterface& work) {
  if (!enabled_)
    return;
  if (work.startTraceUpdated_)
    return;

  work.startTraceUpdated_ = true;
  // If not successful, disable the debugger
  enabled_ = c10d::traceUpdate(
      store_, traceKeyStart_, work.seq_, opTypeToString(work.getOpType()));
}

// Log work end to store.
void ProcessGroupInterface::DesyncDebugger::logWorkEnd(ProcessGroupInterface::WorkInterface& work) {
  if (!enabled_)
    return;

  // In case the start of the work hasn't been logged
  if (!work.startTraceUpdated_) {
    logWorkStart(work);
  }

  // If not successful, disable the debugger
  enabled_ = c10d::traceUpdate(
      store_, traceKeyEnd_, work.seq_, opTypeToString(work.getOpType()));
}


} // namespace c10d

#endif // USE_C10D_XCCL
