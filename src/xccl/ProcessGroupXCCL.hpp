#pragma once

#ifdef USE_C10D_XCCL
// We will define those flags in XCCL backend file instead of passing to gcc
// compiler.
#define CCL_ENABLE_ZE
#define CCL_ENABLE_SYCL

#include <oneapi/ccl.hpp>
#include <exception>
#include <future>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <ATen/xpu/XPUEvent.h>
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/TraceUtils.h>
#include <torch/csrc/distributed/c10d/logger.hpp>
namespace c10d {

static std::vector<std::string> TORCH_XCCL_BLOCKING_WAIT = {
    "TORCH_XCCL_BLOCKING_WAIT",
    "XCCL_BLOCKING_WAIT"};

static std::vector<std::string> TORCH_XCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN = {
    "TORCH_XCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN",
    "XCCL_LOG_CPP_STACK_ON_UNCLEAN_SHUTDOWN"};

static std::vector<std::string> TORCH_XCCL_TRACE_BUFFER_SIZE = {
    "TORCH_XCCL_TRACE_BUFFER_SIZE",
    "XCCL_TRACE_BUFFER_SIZE"};

static std::vector<std::string> TORCH_XCCL_COORD_CHECK_MILSEC = {
    "TORCH_XCCL_COORD_CHECK_MILSEC",
    "XCCL_COORD_CHECK_MILSEC"};

static std::vector<std::string> TORCH_XCCL_DEBUG_INFO_PIPE_FILE = {
    "TORCH_XCCL_DEBUG_INFO_PIPE_FILE",
    "XCCL_DEBUG_INFO_PIPE_FILE"};

#if defined(__linux__)
struct DumpPipe {
  DumpPipe(int rank) {
    std::string fileStem =
        getCvarString(TORCH_XCCL_DEBUG_INFO_PIPE_FILE, "");
    if (fileStem.empty() ||
        getCvarInt(TORCH_XCCL_TRACE_BUFFER_SIZE, 0) <= 0) {
      return;
    }
    TORCH_CHECK(!fileStem.empty(), "TORCH_XCCL_DEBUG_INFO_PIPE_FILE is empty");
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
              << " has been opened, write to it to trigger XCCL Debug Dump.";
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

using xcclComm_t = ccl::communicator;
constexpr const char* XCCL_BACKEND_NAME = "xccl";

class TensorShelf {
 public:
  void stash(std::vector<at::Tensor>& tensors);

  void stash(TensorShelf& other);

  void unstash();

  bool empty();

  void clear();

 protected:
  std::vector<at::Tensor>& get();

 private:
  std::vector<at::Tensor> tVector_;

  std::mutex mutex_;
};

class TORCH_API ProcessGroupXCCL : public Backend {
 public:
  class WorkXCCL : public Work {
   public:
    WorkXCCL(
        at::Device& device,
        int rank,
        OpType opType,
        uint64_t seq,
        bool isP2P,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt);
    WorkXCCL(const WorkXCCL& w);
    ~WorkXCCL() override;

    bool isCompleted() override;

    void abort() override {
      TORCH_CHECK(false, "ProcessGroupXCCL::WorkXCCL::abort not implemented");
    }

    void synchronize() override;

    void synchronizeStream();

    bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override {
      return future_;
    }

    uint64_t getSequencenumber() const override {
      return seq_;
    }

    std::vector<at::Tensor> result() override {
      return *outputs_;
    }

   protected:
    at::Device device_;
    std::shared_ptr<at::xpu::XPUEvent> xcclStartEvent_;
    std::shared_ptr<at::xpu::XPUEvent> xcclEndEvent_;
    bool isBarrierOp_{false};
    bool blockingWait_{false};
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;
    uint64_t seq_;
    bool isP2P_;
    std::optional<uint64_t> trace_id_;
    size_t numelIn_ = -1;
    size_t numelOut_ = -1;

   private:
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
    std::shared_ptr<TensorShelf> stashed_for_allocator_safety_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupXCCL;
  };

  struct Options : public Backend::Options {
    explicit Options();

    static c10::intrusive_ptr<Options> create() {
      return c10::make_intrusive<Options>();
    }

    std::vector<uint64_t> global_ranks_in_group;
    std::string group_name;
  };

  class HeartbeatMonitor {
   public:
    HeartbeatMonitor(ProcessGroupXCCL* pg);
    virtual ~HeartbeatMonitor() = default;

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

  ProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options = Options::create());

  C10_DEPRECATED ProcessGroupXCCL(
      const c10::intrusive_ptr<Store>& store,
      int rank,
      int size,
      const std::string& groupName)
      : ProcessGroupXCCL(store, rank, size) {}

  ~ProcessGroupXCCL() override;

  const std::string getBackendName() const override {
    return std::string(XCCL_BACKEND_NAME);
  }

  bool supportsCoalescing() const override {
    return true;
  }

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing(OpType optype);

  std::shared_ptr<xcclComm_t> getXCCLComm(
      const std::string& deviceKey,
      at::Device& device,
      OpType opType,
      int p2pRank = 0,
      bool isSendRecvSelf = false);

  virtual c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> initWork(
      at::Device& device,
      int rank,
      OpType opType,
      bool isP2P,
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {});

  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr) {
    return collective<Fn>(
        input,
        output,
        fn,
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {},
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {},
        opType,
        asyncOp,
        profilingTitle);
  }

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr) {
    auto inputs = std::vector<at::Tensor>{input};
    auto outputs = std::vector<at::Tensor>{output};
    return collective(
        inputs, outputs, fn, pre, post, opType, asyncOp, profilingTitle);
  }

  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr) {
    return collective<Fn>(
        inputs,
        outputs,
        fn,
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {},
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {},
        opType,
        asyncOp,
        profilingTitle);
  }

  template <typename Fn, typename PreProcess, typename PostProcess>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      PreProcess pre,
      PostProcess post,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr);

  template <typename Fn>
  c10::intrusive_ptr<Work> collectiveCoalesced(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType,
      bool asyncOp,
      const char* profilingTitle = nullptr) {
    return collective<Fn>(
        input,
        output,
        fn,
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
          // There are two types of coalesce that require `group_start/end`:
          // 1. **Fast Pass for Operations**: For example,
          // `allreduce_coalesced`. In this case, the backend has control, so
          // the initial group API `ccl::group` is called.
          // 2. **User-Specified Groups**: The user specifies a series of
          // operations as a group in the frontend by calling the coalesce
          // manager. To avoid incorrect judgments of the p2p state, the
          // `xcclActiveGroupCounter_` is introduced to track group calls made
          // in the frontend. In this scenario, the `groupStart` wrap API is
          // used.
          ccl::group_start();
        },
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {
          ccl::group_end();
        },
        opType,
        asyncOp,
        profilingTitle);
  }

  template <typename Fn>
  c10::intrusive_ptr<Work> pointToPoint(
      at::Tensor& tensor,
      Fn fn,
      int peer,
      OpType opType,
      const char* profilingTitle = nullptr);

  c10::intrusive_ptr<Work> allreduce_impl(
      at::Tensor& tensor,
      const char* profilingTitle = "xccl:all_reduce",
      const AllreduceOptions& opts = AllreduceOptions());

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work> _reduce_oop(
      at::Tensor& outputTensors,
      at::Tensor& inputTensors,
      const ReduceOptions& opts = ReduceOptions());

  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work> _broadcast_oop(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const BroadcastOptions& opts);

  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputbuffer,
      at::Tensor& inputbuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  void groupStart();

  void groupEnd();

  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  void setSequenceNumberForGroup() override;

  uint64_t getSequenceNumberForGroup() override;

  std::string createLogPrefix() const;

  const std::string& logPrefix() const;

  c10::DeviceIndex guessDeviceId() const;
  const int& globalRank() const;

  const std::vector<uint64_t>& groupRanks() const;

  void setStartedPgStatus(c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work);

  bool dumpDebuggingInfo(bool includeStackTrace = true);

 protected:
  std::unordered_map<std::string, std::pair<at::xpu::XPUStream, ccl::stream>>
      xcclStreamsMap_;
  std::unordered_map<std::string, at::xpu::XPUEvent> xcclEventsMap_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>> devXCCLCommMap_;
  c10::intrusive_ptr<Store> store_;
  const c10::intrusive_ptr<Options> options_;
  uint64_t xcclCommCounter_{0};
  std::mutex mutex_;
  std::set<int> usedDeviceIdxs_;
  int coalescing_state_ = 0;
  at::Device coalescedDevice_ = at::Device("xpu");
  std::shared_ptr<xcclComm_t> coalescedComm_ = nullptr;
  bool coalescedAsync_;
  TensorShelf coalescedTensors_;
  bool blockingWait_ = false;
  static thread_local uint64_t xcclActiveGroupCounter_;
  uint64_t seqCollective_{0};
  uint64_t seqP2P_{0};
  uint64_t op_id_{0};
  size_t local_id_;
  std::string logPrefix_;
  std::shared_ptr<ProcessGroupStatus> pgStatus_ =
      std::make_shared<ProcessGroupStatus>();
  std::unique_ptr<HeartbeatMonitor> heartbeatMonitor_;
  int traceBufferSize_;

 private:
  std::mutex kvs_mutex;

  ccl::shared_ptr_class<ccl::kvs> get_kvs(
      int rank,
      c10d::Store& store,
      bool singleP2POp = false,
      const std::string& p2pKey = "",
      int p2pRank = 0) {
    std::lock_guard<std::mutex> lock(kvs_mutex);
    ccl::shared_ptr_class<ccl::kvs> kvs;
    std::string storeKey;
    if (!singleP2POp) {
      storeKey = std::to_string(xcclCommCounter_++);
    } else {
      storeKey = p2pKey;
    }
    // Rank 0 broadcast the bootstrap network information to other ranks
    if (rank == 0 || (singleP2POp && p2pRank == 0)) {
      kvs = ccl::create_main_kvs();
      ccl::kvs::address_type main_addr = kvs->get_address();
      auto ccl_kvs_addr =
          std::vector<uint8_t>(main_addr.begin(), main_addr.end());
      store.set(storeKey, ccl_kvs_addr);
    } else {
      auto ccl_kvs_addr = store.get(storeKey);
      if (ccl_kvs_addr.size() != ccl::kvs::address_max_size) {
        throw std::runtime_error("Unexpected ccl kvs addr from the store\n");
      }
      ccl::kvs::address_type main_addr;
      std::copy_n(
          ccl_kvs_addr.begin(), ccl::kvs::address_max_size, main_addr.begin());
      kvs = ccl::create_kvs(main_addr);
    }
    return kvs;
  }
};
} // namespace c10d

namespace {

inline std::string getXcclVersion() {
  auto xccl_version = ccl::get_library_version();
  std::string versionString = std::to_string(xccl_version.major) + "." +
      std::to_string(xccl_version.minor) + "." +
      std::to_string(xccl_version.update);
  return versionString;
}

inline std::string reduceOpToString(c10d::ReduceOp op) {
  switch (op) {
    case c10d::ReduceOp::SUM:
      return "SUM";
    case c10d::ReduceOp::PRODUCT:
      return "PRODUCT";
    case c10d::ReduceOp::MIN:
      return "MIN";
    case c10d::ReduceOp::MAX:
      return "MAX";
    case c10d::ReduceOp::BAND:
      return "BAND";
    case c10d::ReduceOp::BOR:
      return "BOR";
    case c10d::ReduceOp::BXOR:
      return "BXOR";
    case c10d::ReduceOp::AVG:
      return "AVG";
    case c10d::ReduceOp::PREMUL_SUM:
      return "PREMUL_SUM";
    default:
      return "UNKNOWN";
  }
}

// Since the current profiler trace support for XCCL is unclear, wrap
// `RECORD_PARAM_COMMS_DATA` and output parameters as debug logs.
// export TORCH_CPP_LOG_LEVEL=INFO
#define RECORD_PARAM_COMMS_DATA_WITH_LOG(                                    \
    seq,                                                                     \
    pg_name_tuple,                                                           \
    inputTensors,                                                            \
    outputTensors,                                                           \
    rank,                                                                    \
    collective_name,                                                         \
    inNelems,                                                                \
    outNelems,                                                               \
    dType,                                                                   \
    inSplitSizes,                                                            \
    outSplitSizes,                                                           \
    globalRankStart,                                                         \
    globalRankStride,                                                        \
    worldSize,                                                               \
    async_op)                                                                \
  do {                                                                       \
    LOG(INFO) << std::boolalpha << "collective_name: " << collective_name    \
              << ", inNelems: " << inNelems << ", outNelems: " << outNelems  \
              << ", dType: " << dType << ", root/src rank: " << rank         \
              << ", worldSize: " << worldSize << ", async_op: " << async_op; \
    RECORD_PARAM_COMMS_DATA(                                                 \
        seq,                                                                 \
        pg_name_tuple,                                                       \
        inputTensors,                                                        \
        outputTensors,                                                       \
        rank,                                                                \
        collective_name,                                                     \
        inNelems,                                                            \
        outNelems,                                                           \
        dType,                                                               \
        inSplitSizes,                                                        \
        outSplitSizes,                                                       \
        globalRankStart,                                                     \
        globalRankStride,                                                    \
        worldSize);                                                          \
  } while (0)
} // namespace
#endif // USE_C10D_XCCL
