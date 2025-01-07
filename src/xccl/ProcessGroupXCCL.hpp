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
namespace c10d {

static std::vector<std::string> TORCH_XCCL_BLOCKING_WAIT = {
    "TORCH_XCCL_BLOCKING_WAIT",
    "XCCL_BLOCKING_WAIT"};

using xcclComm_t = ccl::communicator;
constexpr const char* XCCL_BACKEND_NAME = "xccl";

class TORCH_API ProcessGroupXCCL : public Backend {
 public:
  class WorkXCCL : public Work {
   public:
    WorkXCCL(
        at::Device& device,
        int rank,
        OpType opType,
        uint64_t seq,
        const char* profilingTitle = nullptr,
        const std::optional<std::vector<at::Tensor>>& inputs = std::nullopt);
    WorkXCCL(const WorkXCCL& w);
    ~WorkXCCL() override;

    bool isCompleted() override;

    void abort() override {
      TORCH_CHECK(false, "ProcessGroupXCCL::WorkXCCL::abort not implemented");
    }

    void synchronize() override;

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
    std::shared_ptr<at::xpu::XPUEvent> xcclEndEvent_;
    at::Tensor barrierTensor_;
    bool blockingWait_ = false;
    std::chrono::time_point<std::chrono::steady_clock> workStartTime_;
    uint64_t seq_;

   private:
    void synchronizeInternal(std::chrono::milliseconds timeout);
    std::shared_ptr<std::vector<at::Tensor>> outputs_;
    c10::intrusive_ptr<at::ivalue::Future> future_;
    friend class ProcessGroupXCCL;
  };

  ProcessGroupXCCL(const c10::intrusive_ptr<Store>& store, int rank, int size);

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
      const char* profilingTitle = nullptr,
      const std::vector<at::Tensor>& inputs = {},
      const std::vector<at::Tensor>& outputs = {});

  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      at::Tensor& input,
      at::Tensor& output,
      Fn fn,
      OpType opType,
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
      const char* profilingTitle = nullptr) {
    auto inputs = std::vector<at::Tensor>{input};
    auto outputs = std::vector<at::Tensor>{output};
    return collective(inputs, outputs, fn, pre, post, opType, profilingTitle);
  }

  template <typename Fn>
  c10::intrusive_ptr<Work> collective(
      std::vector<at::Tensor>& inputs,
      std::vector<at::Tensor>& outputs,
      Fn fn,
      OpType opType,
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
      const char* profilingTitle = nullptr);

  template <typename Fn>
  c10::intrusive_ptr<Work> collectiveCoalesced(
      std::vector<at::Tensor>& input,
      std::vector<at::Tensor>& output,
      Fn fn,
      OpType opType,
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
        profilingTitle);
  }

  c10::intrusive_ptr<Work> allreduce_impl(
      at::Tensor& tensor,
      const AllreduceOptions& opts = AllreduceOptions());

  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

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

  void groupStart();

  void groupEnd();

  void setSequenceNumberForGroup() override;

  uint64_t getSequenceNumberForGroup() override;

 protected:
  std::unordered_map<std::string, at::xpu::XPUStream> xcclStreamsMap_;
  std::unordered_map<std::string, at::xpu::XPUEvent> xcclEventsMap_;
  std::unordered_map<std::string, std::shared_ptr<xcclComm_t>> devXCCLCommMap_;
  c10::intrusive_ptr<Store> store_;
  uint64_t xcclCommCounter_{0};
  std::mutex mutex_;
  std::set<int> usedDeviceIdxs_;
  int coalescing_state_ = 0;
  at::Device coalescedDevice_ = at::Device("xpu");
  std::shared_ptr<xcclComm_t> coalescedComm_ = nullptr;
  bool blockingWait_ = false;
  static thread_local uint64_t xcclActiveGroupCounter_;
  uint64_t seqCollective_{0};
  uint64_t seqP2P_{0};

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
} // namespace
#endif // USE_C10D_XCCL
