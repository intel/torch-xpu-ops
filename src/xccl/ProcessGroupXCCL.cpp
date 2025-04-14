#ifdef USE_C10D_XCCL

#include <comm/XPUGuard.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <xccl/ProcessGroupXCCL.hpp>

namespace c10d {

namespace {

#if defined(CCL_MAJOR_VERSION) &&  \
    ((CCL_MAJOR_VERSION > 2021) || \
     (CCL_MAJOR_VERSION == 2021) && (CCL_MINOR_VERSION >= 15))
#define XCCL_HAS_AVG 1
#endif // oneCCL version >= 2021.15

const std::map<c10d::ReduceOp, onecclRedOp_t> xcclOps = {
    {ReduceOp::MIN, onecclRedOp_t::onecclMin},
    {ReduceOp::MAX, onecclRedOp_t::onecclMax},
    {ReduceOp::SUM, onecclRedOp_t::onecclSum},
    {ReduceOp::PRODUCT, onecclRedOp_t::onecclProd},
#ifdef XCCL_HAS_AVG
    {ReduceOp::AVG, onecclRedOp_t::onecclAvg},
#endif // XCCL_HAS_AVG

};

const std::map<at::ScalarType, onecclDataType_t> xcclDatatypes = {
    {at::kByte, onecclDataType_t::onecclUint8},
    {at::kChar, onecclDataType_t::onecclChar},
    {at::kInt, onecclDataType_t::onecclInt32},
    {at::kLong, onecclDataType_t::onecclInt64},
    {at::kHalf, onecclDataType_t::onecclFloat16},
    {at::kFloat, onecclDataType_t::onecclFloat32},
    {at::kDouble, onecclDataType_t::onecclFloat64},
    {at::kBFloat16, onecclDataType_t::onecclBfloat16},
    {at::kBool, onecclDataType_t::onecclUint8},
    // use for non-reducetion op like allgather
    {at::kFloat8_e5m2, onecclDataType_t::onecclUint8},
    {at::kFloat8_e4m3fn, onecclDataType_t::onecclUint8},
    {at::kFloat8_e4m3fnuz, onecclDataType_t::onecclUint8},
    {at::kFloat8_e5m2fnuz, onecclDataType_t::onecclUint8},
};

bool checkSameSize(const std::vector<at::Tensor>& input_tensors) {
  for (const auto& input_tensor : input_tensors) {
    if (!input_tensors[0].is_same_size(input_tensor)) {
      return false;
    }
  }
  return true;
}

void checkSingleTensor(
    const at::Tensor& tensor,
    const bool p2p = false // whether operation is a P2P operation
) {
  if (!tensor.is_xpu() || tensor.is_sparse()) {
    C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");

    // Skip the following requirements for P2P operations
    if (!tensor.is_contiguous(tensor.suggest_memory_format())) {
      if (p2p) {
        TORCH_WARN_ONCE(
            "Detected non-contiguous tensor in P2P operations. It is user "
            "responsibility to guarantee that source and destination tensors have "
            "the same contiguity format.");
      } else {
        C10_THROW_ERROR(ValueError, "Tensors must be contiguous");
      }
    }
  }
}

int64_t checkTensorOnSameDevice(const std::vector<at::Tensor>& tensors) {
  TORCH_CHECK_WITH(
      ValueError, tensors.size() != 0, "Tensor list must be nonempty");

  const auto& first = tensors.front();

  int64_t total_numel = 0;
  for (const auto& t : tensors) {
    if (!t.is_xpu() || t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be XPU and dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    TORCH_CHECK_WITH(
        ValueError,
        t.get_device() == tensors[0].get_device(),
        "Expected list of tensors on the same device");
    total_numel += t.numel();
  }

  return total_numel;
}

onecclDataType_t getXcclDataType(
    at::ScalarType type,
    bool is_reduction_op = false) {
  if (is_reduction_op)
    TORCH_CHECK(
        !isFloat8Type(type),
        "Float8 dtypes are not currenlty supported for XCCL reductions");
  auto it = xcclDatatypes.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      it != xcclDatatypes.end(),
      "Input tensor data type is not supported for XCCL process group: ",
      type);
  return it->second;
}

onecclRedOp_t getXcclReduceOp(const ReduceOp& reduceOp, at::Tensor& input) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        // Map sum to max for bool tensors to avoid overflow issues with sum.
        return onecclRedOp_t::onecclMax;
      }
#ifdef XCCL_HAS_AVG
      if (reduceOp == ReduceOp::AVG) {
        C10_THROW_ERROR(
            TypeError, "Cannot use ReduceOp.AVG with boolean inputs");
      }
#endif // XCCL_HAS_AVG
    }
#if !defined(XCCL_HAS_AVG)
    if (reduceOp == ReduceOp::AVG) {
      return onecclRedOp_t::onecclSum;
    }
#endif
    return xcclOps.at(reduceOp);
  } catch (const std::out_of_range&) {
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduceOpToString(reduceOp) + " with XCCL");
  }
}

bool complexViewAsRealAllowed(const ReduceOp& reduceOp) {
  switch (reduceOp) {
    case ReduceOp::SUM:
      return true;
    case ReduceOp::AVG:
      return true;
    case ReduceOp::UNUSED:
      return true;
    default:
      return false;
  }
  return false;
}

void syncStream(
    at::Device& device,
    at::xpu::XPUEvent& xcclEvent,
    at::xpu::XPUStream& xcclStream) {
  xcclEvent.record(at::xpu::getCurrentXPUStream(device.index()));
  xcclEvent.block(xcclStream);
}

} // namespace

constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupXCCL::xcclActiveGroupCounter_ = 0;

ProcessGroupXCCL::WorkXCCL::WorkXCCL(
    at::Device& device,
    int rank,
    OpType opType,
    uint64_t seq,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs)
    : Work(rank, opType, profilingTitle, inputs),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq) {
  xcclEndEvent_ = std::make_shared<at::xpu::XPUEvent>();
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(const WorkXCCL& w)
    : Work(w.rank_, w.opType_),
      device_(w.device_),
      xcclEndEvent_(w.xcclEndEvent_),
      blockingWait_(w.blockingWait_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_) {}

ProcessGroupXCCL::WorkXCCL::~WorkXCCL() = default;

bool ProcessGroupXCCL::WorkXCCL::isCompleted() {
  if (xcclEndEvent_ && xcclEndEvent_->query()) {
    return true;
  }
  return false;
}

void ProcessGroupXCCL::WorkXCCL::synchronize() {
  synchronizeInternal(kNoTimeout);
}

void ProcessGroupXCCL::WorkXCCL::synchronizeInternal(
    std::chrono::milliseconds timeout) {
  auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
  xcclEndEvent_->block(currentStream);
  if (blockingWait_) {
    while (!isCompleted()) {
      auto currentTimepoint = std::chrono::steady_clock::now();
      auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          currentTimepoint - workStartTime_);
      if (timeElapsed >= timeout) {
        std::string exceptionMsg = c10::str(
            "Work ran time out after ", timeElapsed.count(), " milliseconds.");
        LOG(ERROR) << exceptionMsg;
        // todo: abort comm and exit
        TORCH_CHECK(false, exceptionMsg)
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    }
  }
  if (barrierTensor_.defined()) {
    auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
    currentStream.synchronize();
  }
}

bool ProcessGroupXCCL::WorkXCCL::wait(std::chrono::milliseconds timeout) {
  synchronizeInternal(timeout);
  return true;
}

static std::atomic<size_t> process_group_id = 0;

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple";

std::string ProcessGroupXCCL::createLogPrefix() const {
  if (!pg_desc_.empty() && pg_desc_ != "undefined") {
    return c10::str(
        "[PG ID ",
        local_id_,
        " PG GUID ",
        pg_uid_,
        "(",
        pg_desc_,
        ") Rank ",
        rank_,
        "] ");
  }
  return c10::str(
      "[PG ID ", local_id_, " PG GUID ", pg_uid_, " Rank ", rank_, "] ");
}

const std::string& ProcessGroupXCCL::logPrefix() const {
  return logPrefix_;
}

ProcessGroupXCCL::ProcessGroupXCCL(
    c10::intrusive_ptr<Store> store,
    int rank,
    int size,
    c10::intrusive_ptr<Options> options)
    : Backend(rank, size),
      store_(std::move(store)),
      options_(std::move(options)),
      xcclCommCounter_(0),
      local_id_(process_group_id++) {
  this->setGroupUid(options_->group_name);
  logPrefix_ = createLogPrefix();
  blockingWait_ = getCvarBool(TORCH_XCCL_BLOCKING_WAIT, false);
  init();
  const std::string OFF = "OFF";
  std::string torch_distributed_debug =
      getCvarString({"TORCH_DISTRIBUTED_DEBUG"}, OFF.c_str());
  const auto XcclVersion = getXcclVersion();
  LOG(INFO) << logPrefix() << "ProcessGroupXCCL initialization options: "
            << "size: " << size << ", global rank: " << rank_;

  LOG(INFO) << logPrefix() << "ProcessGroupXCCL environments: "
            << "XCCL version: " << XcclVersion
            << ", TORCH_XCCL_BLOCKING_WAIT: " << blockingWait_
            << ", TORCH_DISTRIBUTED_DEBUG: " << torch_distributed_debug;
}

ProcessGroupXCCL::~ProcessGroupXCCL() = default;

void ProcessGroupXCCL::setSequenceNumberForGroup() {}

uint64_t ProcessGroupXCCL::getSequenceNumberForGroup() {
  return seqCollective_;
}

c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> ProcessGroupXCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    const char* profilingTitle,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs) {
  auto r = c10::make_intrusive<ProcessGroupXCCL::WorkXCCL>(
      device,
      rank,
      opType,
      seqCollective_,
      profilingTitle,
      std::optional<std::vector<at::Tensor>>(inputs));
  return r;
}

void ProcessGroupXCCL::broadcastUniqueXCCLID(
    onecclUniqueId* xcclID,
    bool isSingleP2POp,
    const std::string& p2pKey,
    int p2pRank) {
  std::string storeKey;
  if (!isSingleP2POp) {
    storeKey = std::to_string(xcclCommCounter_++);
  } else {
    storeKey = p2pKey;
  }
  if (rank_ == 0 || (isSingleP2POp && p2pRank == 0)) {
    auto vec = std::vector<uint8_t>(
        reinterpret_cast<uint8_t*>(xcclID),
        reinterpret_cast<uint8_t*>(xcclID) + ONECCL_UNIQUE_ID_BYTES);
    store_->set(storeKey, vec);
  } else {
    try {
      auto vec = store_->get(storeKey);
      TORCH_CHECK_WITH(
          DistBackendError,
          vec.size() == ONECCL_UNIQUE_ID_BYTES,
          "Invalid size for xcclUniqueId");
      std::memcpy(xcclID, vec.data(), vec.size());
    } catch (const std::exception& e) {
      std::string exceptionMsg = c10::str(
          "[",
          rank_,
          "] is setting up XCCL communicator and "
          "retrieving xcclUniqueId from [0] via c10d key-value store by key '",
          storeKey,
          "', but store->get('",
          storeKey,
          "') got error: ");
      C10_THROW_ERROR(
          DistBackendError,
          exceptionMsg + e.what() +
              ". This may indicate a possible application crash on rank 0 or a network set up issue.");
    } catch (...) {
      C10_THROW_ERROR(
          DistBackendError,
          c10::str(
              "Unknown exception while [",
              rank_,
              "] is setting up XCCL communicator and "
              "retrieving xcclUniqueId from [0] via c10d key-value store by key '",
              storeKey,
              "'",
              ". This may indicate a possible application crash on rank 0 or a network set up issue."));
    }
  }
}

std::shared_ptr<xcclComm_t> ProcessGroupXCCL::getXCCLComm(
    const std::string& deviceKey,
    at::Device& device,
    OpType opType,
    int p2pRank,
    bool isSendRecvSelf) {
  if (deviceKey.empty()) {
    C10_THROW_ERROR(
        DistBackendError,
        "Not able to create/get the XCCL Communicator since "
        "the devices are empty ");
  }

  usedDeviceIdxs_.insert(device.index());

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (devXCCLCommMap_.find(deviceKey) != devXCCLCommMap_.end()) {
      return devXCCLCommMap_[deviceKey];
    }
  }

  std::shared_ptr<xcclComm_t> XCCLComm;
  onecclUniqueId xcclID;

  bool batchP2P = xcclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);

  at::xpu::OptionalXPUGuard gpuGuard(device);

  for (const auto i : c10::irange(xcclActiveGroupCounter_)) {
    (void)i;
    onecclGroupEnd();
  }

  int numRanks, rank;
  if (!singleP2POp) {
    numRanks = getSize();
    rank = getRank();
  } else if (isSendRecvSelf) {
    numRanks = 1;
    rank = 0;
  } else {
    numRanks = 2;
    rank = p2pRank;
  }

  bool force_high = getCvarBool(TORCH_XCCL_HIGH_PRIORITY, false);
  c10::Stream stream = at::xpu::getStreamFromPool(
      options_->is_high_priority_stream || force_high);

  if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
    onecclGetUniqueId(&xcclID);
  }
  broadcastUniqueXCCLID(&xcclID, singleP2POp, deviceKey, p2pRank);

  xcclComm_t comm = nullptr;
  onecclResult_t result = onecclSuccess;
  result = onecclSetDevice(rank);
  if (result != onecclSuccess) {
    std::cerr << "Failed to set device.\n";
  }
  result = onecclCommInitRank(&comm, numRanks, xcclID, rank);
  if (result != onecclSuccess) {
    std::cerr << "Failed to initialize communicator.\n";
  }
  XCCLComm = std::make_shared<xcclComm_t>(comm);

  RECORD_PARAM_COMMS(
      0, // seq
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank, // rank
      "init", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      size_); // worldSize

  for (const auto i : c10::irange(xcclActiveGroupCounter_)) {
    (void)i;
    onecclGroupStart();
  }

  // The oneCCL group API requires retaining the SYCL queue (SyclQueue) object
  // within the lifecycle of the communicator. If the XPU stream is created
  // within the collective operation, it would be destroyed earlier than the
  // communicator after the operation ends. Therefore, the XPU stream is stored
  // in a map alongside the communicator. Similarly, oneCCLv2 also requires
  // retaining the SYCL queue pointer for collective operations, so this change
  // will be necessary in oneCCLv2 as well.
  std::lock_guard<std::mutex> lock(mutex_);
  sycl::queue& q = c10::xpu::XPUStream(stream).queue();
  devXCCLCommMap_.emplace(deviceKey, XCCLComm);
  xcclStreamsMap_.emplace(
      deviceKey, std::make_pair(at::xpu::XPUStream(stream), q));
  xcclEventsMap_.emplace(deviceKey, at::xpu::XPUEvent());

  LOG(INFO) << logPrefix()
            << "Created XCCL communicator with Key: " << deviceKey;

  return XCCLComm;
}

void ProcessGroupXCCL::groupStart() {
  onecclGroupStart();
  ++xcclActiveGroupCounter_;
}

void ProcessGroupXCCL::groupEnd() {
  onecclGroupEnd();
  --xcclActiveGroupCounter_;
}

ProcessGroupXCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(XCCL_BACKEND_NAME),
      is_high_priority_stream(is_high_priority_stream) {}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;
void ProcessGroupXCCL::startCoalescing() {
  if (coalescing_state_ & CoalP2P) {
    seqP2P_++;
  } else {
    seqCollective_++;
  }
  coalescedDevice_.set_index(-1);
  coalescedComm_ = nullptr;
  coalescing_state_ |= CoalActive;
  groupStart();
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::endCoalescing(OpType optype) {
  if (coalescedComm_ == nullptr) {
    // There is no actual work being coalesced, return here
    groupEnd();
    coalescing_state_ = 0;
    return nullptr;
  }
  TORCH_CHECK(
      coalescedDevice_.index() >= 0,
      "Somthing went wrong. Did you call end_coalescing before start_coalescing?");

  auto comm = coalescedComm_;
  auto device = coalescedDevice_;

  const auto key = std::to_string(device.index());
  auto stream = xcclStreamsMap_.at(key).first;

  auto work = initWork(device, rank_, optype);
  work->blockingWait_ = blockingWait_;

  groupEnd();

  work->xcclEndEvent_->record(stream);

  coalescing_state_ = 0;
  coalescedComm_ = nullptr;
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::endCoalescing() {
  // Default OpType to COALESCED if not specified
  return endCoalescing(OpType::COALESCED);
}

template <typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<Work> ProcessGroupXCCL::collective(
    std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    Fn fn,
    PreProcess pre,
    PostProcess post,
    OpType opType,
    const char* profilingTitle) {
  seqCollective_++;
  auto device = inputs[0].device();
  const auto key = std::to_string(device.index());
  auto comm = getXCCLComm(key, device, opType);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalColl;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = comm;
    } else {
      TORCH_CHECK(coalescedComm_ == comm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  auto stream = xcclStreamsMap_.at(key).first;
  auto cclstream = xcclStreamsMap_.at(key).second;
  syncStream(device, xcclEventsMap_[key], stream);

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
  work = initWork(device, rank_, opType);

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  at::xpu::OptionalXPUGuard gpuGuard(device);

  pre(stream, work);

  for (const auto i : c10::irange(inputs.size())) {
    c10::xpu::XPUCachingAllocator::recordStream(
        inputs[i].storage().data_ptr(), stream);
    fn(inputs[i], outputs[i], *comm, stream, cclstream);
  }

  post(stream, work);

  if (!coalescing_state_) {
    work->xcclEndEvent_->record(stream);
  }

  std::vector<c10::Stream> streams = {stream.unwrap()};
  c10::MultiStreamGuard streamGuard(streams);
  std::vector<at::Device> devices{device};
  work->future_ = c10::make_intrusive<at::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(at::IValue(*work->outputs_));
  work->blockingWait_ = blockingWait_;

  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work> ProcessGroupXCCL::pointToPoint(
    at::Tensor& tensor,
    Fn fn,
    int peer,
    OpType opType,
    const char* profilingTitle) {
  auto device = tensor.device();
  std::string key;
  int p2pRank = 0, p2pTargetRank = 0;
  bool isSendRecvSelf = false;

  bool batchP2P = xcclActiveGroupCounter_ > 0;
  if (batchP2P) {
    key = std::to_string(device.index());
    p2pRank = rank_;
    p2pTargetRank = peer;
  } else {
    int lowRank = rank_ < peer ? rank_ : peer;
    int highRank = rank_ < peer ? peer : rank_;
    key = std::to_string(lowRank) + ":" + std::to_string(highRank);
    p2pRank = rank_ <= peer ? 0 : 1;
    isSendRecvSelf = rank_ == peer;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;
    if (!coalescing_state_) {
      seqP2P_++;
    }
  }

  auto comm = getXCCLComm(key, device, opType, p2pRank, isSendRecvSelf);

  if (coalescing_state_ & CoalActive) {
    coalescing_state_ |= CoalP2P;
    if (coalescedDevice_.index() < 0) {
      coalescedDevice_ = device;
    } else {
      TORCH_CHECK(
          coalescedDevice_.index() == device.index(), MULTI_DEVICE_ERROR_MSG);
    }
    if (coalescedComm_ == nullptr) {
      coalescedComm_ = comm;
    } else {
      TORCH_CHECK(coalescedComm_ == comm, MULTI_DEVICE_ERROR_MSG);
    }
  }

  auto stream = xcclStreamsMap_.at(key).first;
  auto cclstream = xcclStreamsMap_.at(key).second;
  syncStream(device, xcclEventsMap_[key], stream);

  if (!coalescing_state_) {
    c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
    work = initWork(device, rank_, opType);
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>();
    work->outputs_->push_back(tensor);

    at::xpu::OptionalXPUGuard gpuGuard(device);

    c10::xpu::XPUCachingAllocator::recordStream(
        tensor.storage().data_ptr(), stream);

    fn(tensor, *comm, stream, cclstream, p2pTargetRank);

    work->xcclEndEvent_->record(stream);
    work->blockingWait_ = blockingWait_;
    std::vector<c10::Stream> streams = {stream.unwrap()};
    c10::MultiStreamGuard streamGuard(streams);
    std::vector<at::Device> devices{device};
    work->future_ = c10::make_intrusive<at::ivalue::Future>(
        c10::ListType::create(c10::TensorType::get()), devices);
    work->future_->markCompleted(at::IValue(*work->outputs_));
    return work;
  } else {
    at::xpu::OptionalXPUGuard gpuGuard(device);

    c10::xpu::XPUCachingAllocator::recordStream(
        tensor.storage().data_ptr(), stream);

    fn(tensor, *comm, stream, cclstream, p2pTargetRank);

    return nullptr;
  }
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  checkSingleTensor(tensor, true);

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      dstRank, // dst rank
      "send", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& input,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue,
          int dst) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        onecclSend(
            input.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            dst,
            comm,
            &SyclQueue);
        return;
      },
      dstRank,
      OpType::SEND,
      c10::str("xccl:send ", rank_, "->", dstRank).c_str());
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int /* unused */) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto tensor = tensors.back();
  checkSingleTensor(tensor, true);

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      srcRank, // src rank
      "recv", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue,
          int src) {
        auto xcclDataType = getXcclDataType(output.scalar_type());
        onecclRecv(
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            src,
            comm,
            &SyclQueue);
        return;
      },
      srcRank,
      OpType::RECV,
      c10::str("xccl:recv ", rank_, "<-", srcRank).c_str());
  return ret;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupXCCL::gather: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto inputTensor = inputTensors.back();

  std::vector<at::Tensor> outputs;

  if (getRank() == opts.rootRank) {
    if (outputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element output list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (outputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect output list size " << outputTensors[0].size()
         << ". Output list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = inputTensor.options();
    const auto& sizes = inputTensor.sizes();
    assertTypeAndSizesMatch(invalidArgument, outputTensors[0], options, sizes);
    outputs = outputTensors[0];
  } else {
    // if not in the root rank, initialize outputs as empty list
    if (outputTensors.size() != 0) {
      invalidArgument("requires empty output on non-root");
    }
    outputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    outputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      opts.rootRank, // root rank
      "gather", // collective name
      inputTensor.numel(), // inNelems
      inputTensor.numel() * this->getSize(), // outNelems
      inputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  auto inputs = std::vector<at::Tensor>{inputTensor};
  return collective(
      inputs,
      outputs, // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        const auto root = opts.rootRank;
        if (getRank() == root) {
          for (auto output : outputs) {
            c10::xpu::XPUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
          }
        }
        {
          auto xcclDataType = getXcclDataType(inputTensor.scalar_type());
          if (rank_ == root) {
            for (const auto r : c10::irange(size_)) {
              if (r != root) {
                // do receive
                onecclRecv(
                    outputs[r].data_ptr(),
                    (size_t)inputTensor.numel(),
                    xcclDataType,
                    r,
                    comm,
                    &SyclQueue);
              } else {
                // on its own rank, simply copy from the input
                outputs[r].copy_(inputTensor);
              }
            }
          } else {
            // do send
            onecclSend(
                inputTensor.data_ptr(),
                (size_t)inputTensor.numel(),
                xcclDataType,
                root,
                comm,
                &SyclQueue);
          }
          return;
        }
      },
      OpType::GATHER,
      "xccl:gather");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupXCCL::scatter: " + msg);
  };

  assertRootRank(invalidArgument, opts.rootRank, size_);

  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto outputTensor = outputTensors.back();

  std::vector<at::Tensor> inputs;

  if (getRank() == opts.rootRank) {
    if (inputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with "
         << getSize() << " tensors.";
      invalidArgument(ss.str());
    } else if (inputTensors[0].size() != static_cast<size_t>(getSize())) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputTensors[0].size()
         << ". Input list size should be " << getSize()
         << ", same as size of the process group.";
      invalidArgument(ss.str());
    }

    const auto& options = outputTensor.options();
    const auto& sizes = outputTensor.sizes();
    assertTypeAndSizesMatch(invalidArgument, inputTensors[0], options, sizes);
    inputs = inputTensors[0];
  } else {
    // if not in the root rank, initialize inputTensors as empty place
    // holder with an empty list
    if (inputTensors.size() != 0) {
      invalidArgument("requires empty input on non-root");
    }
    inputs = {};
    // append a empty tensor to the list, we don't use it but the
    // `collective` template function requires it to invoke its function
    inputs.emplace_back();
  }

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      opts.rootRank, // root rank
      "scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  const auto root = opts.rootRank;

  auto outputs = std::vector<at::Tensor>{outputTensor};
  return collective(
      outputs,
      inputs, // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        if (getRank() == root) {
          for (auto input : inputs) {
            c10::xpu::XPUCachingAllocator::recordStream(
                input.storage().data_ptr(), stream);
          }
        }
        {
          if (rank_ == root) {
            for (const auto r : c10::irange(size_)) {
              if (r != root) {
                // do send
                size_t send_count = inputs[r].numel();
                auto send_type = getXcclDataType(inputs[r].scalar_type());
                onecclSend(
                    inputs[r].data_ptr(),
                    send_count,
                    send_type,
                    r,
                    comm,
                    &SyclQueue);
              } else {
                // on its own rank, simply copy from the input
                outputTensor.copy_(inputs[r]);
              }
            }
          } else {
            // do receive
            size_t recv_count = outputTensor.numel();
            auto recv_type = getXcclDataType(outputTensor.scalar_type());
            onecclRecv(
                outputTensor.data_ptr(),
                recv_count,
                recv_type,
                root,
                comm,
                &SyclQueue);
          }

          return;
        }
      },
      OpType::SCATTER,
      "xccl:scatter");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce_impl(
    at::Tensor& tensor,
    const char* profilingTitle,
    const AllreduceOptions& opts) {
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        onecclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            &SyclQueue);
#if !defined(XCCL_HAS_AVG)
        if (opts.reduceOp == ReduceOp::AVG) {
          auto divisor = getSize();
          c10::StreamGuard guard(stream);
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          output.div_(divisor);
        }
#endif
        return;
      },
      OpType::ALLREDUCE,
      profilingTitle);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(
        complexViewAsRealAllowed(opts.reduceOp),
        "all_reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  checkSingleTensor(tensor);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      size_); // worldSize

  return allreduce_impl(tensor, "xccl:all_reduce", opts);
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  auto total_numel = checkTensorOnSameDevice(tensors);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      rank_, // rank
      "allreduce_coalesced", // collective name
      total_numel, // inNelems
      total_numel, // outNelems
      tensors[0].scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  return collectiveCoalesced(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        onecclAllReduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            &SyclQueue);
#if !defined(XCCL_HAS_AVG)
        if (opts.reduceOp == ReduceOp::AVG) {
          auto divisor = getSize();
          c10::StreamGuard guard(stream);
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          output.div_(divisor);
        }
#endif
        return;
      },
      OpType::COALESCED,
      "xccl:allreduce_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    tensor = at::view_as_real(tensor);
  }
  checkSingleTensor(tensor);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      opts.rootRank, // root rank
      "broadcast", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  const auto root = opts.rootRank + opts.rootTensor;

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        onecclBroadcast(
            input.data_ptr(),
            output.data_ptr(), // ?
            (size_t)input.numel(),
            xcclDataType,
            root,
            comm,
            &SyclQueue);
        return;
      },
      OpType::BROADCAST,
      "xccl:broadcast");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::_broadcast_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const BroadcastOptions& opts) {
  if (outputTensor.numel() != inputTensor.numel()) {
    C10_THROW_ERROR(
        ValueError,
        "Tensor input and output of _broadcast_oop must have the same number of elements ");
  }
  const auto root = opts.rootRank + opts.rootTensor;
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        onecclBroadcast(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            root,
            comm,
            &SyclQueue);
        return;
      },
      OpType::BROADCAST,
      "xccl:_broadcast_oop");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  if (tensor.is_complex()) {
    TORCH_CHECK(
        complexViewAsRealAllowed(opts.reduceOp),
        "reduce does not support",
        opts.reduceOp,
        "on complex tensors");
    tensor = at::view_as_real(tensor);
  }
  checkSingleTensor(tensor);

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      tensors, // inputTensors
      tensors, // outputTensors
      opts.rootRank, // root rank
      "reduce", // collective name
      tensor.numel(), // inNelems
      tensor.numel(), // outNelems
      tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        const int root = opts.rootRank + opts.rootTensor;
        const auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        const auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        onecclReduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            root,
            comm,
            &SyclQueue);
#if !defined(XCCL_HAS_AVG)
        if (opts.reduceOp == ReduceOp::AVG && getRank() == root) {
          auto divisor = getSize();
          c10::StreamGuard guard(stream);
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          output.div_(divisor);
        }
#endif
        return;
      },
      OpType::REDUCE,
      "xccl:reduce");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::_reduce_oop(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceOptions& opts) {
  TORCH_CHECK_WITH(
      ValueError,
      outputTensor.numel() == inputTensor.numel(),
      "Tensor input and output of _reduce_oop must have the same number of elements");
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        const int root = opts.rootRank + opts.rootTensor;
        const auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        const auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        onecclReduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            root,
            comm,
            &SyclQueue);
#if !defined(XCCL_HAS_AVG)
        if (opts.reduceOp == ReduceOp::AVG && getRank() == root) {
          auto divisor = getSize();
          c10::StreamGuard guard(stream);
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          output.div_(divisor);
        }
#endif
        return;
      },
      OpType::REDUCE,
      "xccl:_reduce_oop");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  TORCH_CHECK(inputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto inputTensor = inputTensors.back();
  checkSingleTensor(inputTensor);
  // @lint-ignore CLANGTIDY
  std::vector<at::Tensor>& outputTensors_ = outputTensors.back();

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "all_gather", // collective name
      inputTensor.numel(), // inNelems
      inputTensor.numel() * // outNelems
          this->getSize(),
      inputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  bool same_size = checkSameSize(outputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor outputFlattened = newLikeFlat(outputTensors_);

    return collective(
        inputTensor,
        outputFlattened,
        [&](at::Tensor& input,
            at::Tensor& output,
            xcclComm_t& comm,
            at::xpu::XPUStream& stream,
            sycl::queue& SyclQueue) {
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          auto xcclDataType = getXcclDataType(input.scalar_type());
          onecclAllGather(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)input.numel(),
              xcclDataType,
              comm,
              &SyclQueue);
          return;
        },
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
        [&](at::xpu::XPUStream& Stream,
            c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {
          // Copy the flattened output tensors to the outputs.
          c10::StreamGuard guard(Stream);
          for (const auto j : c10::irange(outputTensors_.size())) {
            c10::xpu::XPUCachingAllocator::recordStream(
                outputTensors_[j].storage().data_ptr(), Stream);
            outputTensors_[j].copy_(outputFlattened[j], true);
          }
        },
        OpType::ALLGATHER,
        "xccl:all_gather");
  } else {
    const auto num_reduces = outputTensors_.size();
    startCoalescing();
    for (const int i : c10::irange(num_reduces)) {
      auto& output = outputTensors_[i];
      auto& input = (i == rank_) ? inputTensor : output;
      auto broadcastOpts = BroadcastOptions{
          static_cast<int64_t>(i), static_cast<int64_t>(0), opts.timeout};
      _broadcast_oop(output, input, broadcastOpts);
    }
    auto work = endCoalescing(OpType::ALLGATHER);
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::_allgather_base(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const AllgatherOptions& opts) {
  checkSingleTensor(input_tensor);
  checkSingleTensor(output_tensor);

  TORCH_CHECK_WITH(
      TypeError,
      input_tensor.dtype() == output_tensor.dtype(),
      "output tensor must have the same type as input tensor");
  TORCH_CHECK_WITH(
      ValueError,
      input_tensor.numel() * size_ == output_tensor.numel(),
      "output tensor size must be equal to world_size times input tensor size");

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      input_tensor, // inputTensors
      output_tensor, // outputTensors
      rank_, // rank
      "_allgather_base", // collective name
      input_tensor.numel(), // inNelems
      output_tensor.numel(), // outNelems
      output_tensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSize
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  return collective(
      input_tensor,
      output_tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        auto xcclDataType = getXcclDataType(input.scalar_type());
        onecclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            comm,
            &SyclQueue);
        return;
      },
      OpType::_ALLGATHER_BASE,
      "xccl:_all_gather_base");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective and assume only one collective
                  // in coalesed range
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputs, // inputTensors
      outputs, // outputTensors
      rank_, // rank
      "allgather_into_tensor_coalesced", // collective name
      getTensorsNumel(inputs), // inNelems
      getTensorsNumel(outputs), // outNelems
      inputs[0].scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        onecclAllGather(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            comm,
            &SyclQueue);
        return;
      },
      OpType::COALESCED,
      "xccl:all_gather_into_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK(outputTensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  // @lint-ignore CLANGTIDY
  auto outputTensor = outputTensors.back();
  checkSingleTensor(outputTensor);
  // @lint-ignore CLANGTIDY
  auto inputTensors_ = inputTensors.back();

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "reduce_scatter", // collective name
      outputTensor.numel() * this->getSize(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  bool same_size = checkSameSize(inputTensors_);
  if (same_size) {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor inputFlattened = newLikeFlat(inputTensors_);
    return collective(
        inputFlattened,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            xcclComm_t& comm,
            at::xpu::XPUStream& stream,
            sycl::queue& SyclQueue) {
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          auto xcclDataType = getXcclDataType(input.scalar_type(), true);
          auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
          onecclReduceScatter(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)output.numel(),
              xcclDataType,
              xcclReduceOp,
              comm,
              &SyclQueue);
#if !defined(XCCL_HAS_AVG)
          if (opts.reduceOp == ReduceOp::AVG) {
            auto divisor = getSize();
            c10::StreamGuard guard(stream);
            c10::xpu::XPUCachingAllocator::recordStream(
                output.storage().data_ptr(), stream);
            output.div_(divisor);
          }
#endif
          return;
        },
        [&](at::xpu::XPUStream& Stream,
            c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {
          // Copy the input tensors to the flattened inputs.
          c10::StreamGuard guard(Stream);
          for (const auto j : c10::irange(inputTensors_.size())) {
            c10::xpu::XPUCachingAllocator::recordStream(
                inputTensors_[j].storage().data_ptr(), Stream);
            inputFlattened[j].copy_(inputTensors_[j], true);
          }
        },
        [&](at::xpu::XPUStream&,
            c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {},
        OpType::REDUCE_SCATTER,
        "xccl:reduce_scatter");
  } else {
    const auto num_reduces = inputTensors_.size();
    startCoalescing();
    for (const int i : c10::irange(num_reduces)) {
      auto& input = inputTensors_[i];
      auto& output = (i == rank_) ? outputTensor : input;
      auto reduceOpts = ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i),
          static_cast<int64_t>(0),
          opts.timeout};
      _reduce_oop(output, input, reduceOpts);
    }
    auto work = endCoalescing(OpType::REDUCE_SCATTER);
    return work;
  }
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::_reduce_scatter_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    const ReduceScatterOptions& opts) {
  TORCH_CHECK_WITH(
      TypeError,
      inputTensor.dtype() == outputTensor.dtype(),
      "input tensor must be the same type as the output tensor.");
  TORCH_CHECK_WITH(
      ValueError,
      inputTensor.numel() == outputTensor.numel() * size_,
      "input tensor must be the same size as output size times world size");

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensor, // inputTensor
      outputTensor, // outputTensor
      rank_, // rank
      "_reduce_scatter_base", // collective name
      inputTensor.numel(), // inNelems
      outputTensor.numel(), // outNelems
      outputTensor.scalar_type(), // dtype
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        onecclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            &SyclQueue);
#if !defined(XCCL_HAS_AVG)
        if (opts.reduceOp == ReduceOp::AVG) {
          auto divisor = getSize();
          c10::StreamGuard guard(stream);
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          output.div_(divisor);
        }
#endif
        return;
      },
      OpType::_REDUCE_SCATTER_BASE,
      "xccl:_reduce_scatter_base");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        onecclReduceScatter(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            &SyclQueue);
#if !defined(XCCL_HAS_AVG)
        if (opts.reduceOp == ReduceOp::AVG) {
          auto divisor = getSize();
          c10::StreamGuard guard(stream);
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          output.div_(divisor);
        }
#endif
        return;
      },
      OpType::COALESCED,
      "xccl:reduce_scatter_tensor_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::barrier(const BarrierOptions& opts) {
  RECORD_PARAM_COMMS(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      rank_, // rank
      "barrier", // collective name
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize
  // Device to use for barrier
  int barDevIdx = -1;

  // See nccl barrier comments
  if (!opts.device_ids.empty()) {
    barDevIdx = opts.device_ids[0];
  } else if (getBoundDeviceId()) {
    barDevIdx = (*getBoundDeviceId()).index();
  } else if (!usedDeviceIdxs_.empty()) {
    barDevIdx = *usedDeviceIdxs_.begin();
  } else {
    barDevIdx =
        static_cast<int16_t>(rank_ % at::detail::getXPUHooks().getNumGPUs());
  }

  TORCH_CHECK_WITH(
      ValueError,
      barDevIdx >= 0,
      "Failed to infer a GPU device id to perform barrier. ");
  auto barDevice = at::Device(at::DeviceType::XPU, barDevIdx);

  at::Tensor barrierTensor =
      at::zeros({1}, at::TensorOptions().device(barDevice).dtype(at::kFloat));

  auto work = allreduce_impl(barrierTensor, "xccl:all_reduce_barrier");

  auto xcclWork = dynamic_cast<ProcessGroupXCCL::WorkXCCL*>(work.get());
  TORCH_CHECK(xcclWork);
  xcclWork->barrierTensor_ = std::move(barrierTensor);
  return work;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& /* unused */) {
  checkSingleTensor(outputTensor, true);
  checkSingleTensor(inputTensor, true);
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0) {
    RECORD_PARAM_COMMS_DATA_WITH_LOG(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_all", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        std::vector<int64_t>(), // inSplitSizes
        std::vector<int64_t>(), // outSplitSizes
        -1, // globalRankStart
        -1, // globalRankStride
        this->getSize()); // worldSize
    TORCH_CHECK(
        outputTensor.numel() == inputTensor.numel() &&
            outputTensor.scalar_type() == inputTensor.scalar_type(),
        "xpu_alltoall_base: tensors are not equal in size or data type");
    TORCH_CHECK(
        outputTensor.size(0) % size_ == 0,
        "xpu_alltoall_base: tensor's dim 0 does not divide equally across group size");
  } else {
    c10d::checkSplitSizes(inputSplitSizes, inputTensor, size_);
    c10d::checkSplitSizes(outputSplitSizes, outputTensor, size_);

    RECORD_PARAM_COMMS_DATA_WITH_LOG(
        static_cast<int>(
            this->getSequenceNumberForGroup() +
            1), // seq + 1 to match collective
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        inputTensor, // inputTensor
        outputTensor, // outputTensor
        rank_, // rank
        "all_to_allv", // collective name
        inputTensor.numel(), // inNelems
        outputTensor.numel(), // outNelems
        inputTensor.scalar_type(), // dType
        inputSplitSizes, // inSplitSizes
        outputSplitSizes, // outSplitSizes
        -1, // globalRankStart
        -1, // globalRankStride
        this->getSize()); // worldSize
  }
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        std::vector<size_t> send_lengths(size_);
        std::vector<size_t> recv_lengths(size_);
        std::vector<size_t> send_offsets(size_);
        std::vector<size_t> recv_offsets(size_);
        c10d::computeLengthsAndOffsets(
            inputSplitSizes, input, &send_lengths, &send_offsets);
        c10d::computeLengthsAndOffsets(
            outputSplitSizes, output, &recv_lengths, &recv_offsets);

        size_t size = input.element_size();
        auto xcclDataType = getXcclDataType(input.scalar_type());
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);

        auto send_offsets_data = send_offsets.data();
        auto recv_offsets_data = recv_offsets.data();

        onecclGroupStart();
        for (const auto r : c10::irange(size_)) {
          if (send_lengths[r] != 0) {
            onecclSend(
                ((char*)input.data_ptr()) + send_offsets_data[r] * size,
                send_lengths[r],
                xcclDataType,
                r,
                comm,
                &SyclQueue);
          }
          if (recv_lengths[r] != 0) {
            onecclRecv(
                ((char*)output.data_ptr()) + recv_offsets_data[r] * size,
                recv_lengths[r],
                xcclDataType,
                r,
                comm,
                &SyclQueue);
          }
        }
        onecclGroupEnd();

        return;
      },
      OpType::ALLTOALL_BASE,
      "xccl:all_to_all");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& /* unused */) {
  auto device = outputTensors[0].device();
  int64_t total_numel = 0;
  for (const auto r : c10::irange(outputTensors.size())) {
    checkSingleTensor(outputTensors[r], true);
    checkSingleTensor(inputTensors[r], true);
    TORCH_CHECK(
        device == outputTensors[r].device() &&
            device == inputTensors[r].device(),
        "Tensors must be on the same device")
    total_numel += inputTensors[r].numel();
  }

  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      static_cast<int>(
          this->getSequenceNumberForGroup() + 1), // seq + 1 to match collective
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputTensors, // inputTensors
      outputTensors, // outputTensors
      rank_, // rank
      "all_to_all", // collective name
      total_numel, // inNelems
      total_numel, // outNelems
      inputTensors.front().scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize()); // worldSize

  return collective(
      inputTensors.front(),
      outputTensors.front(),
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          sycl::queue& SyclQueue) {
        onecclGroupStart();
        for (const int r :
             c10::irange(static_cast<int>(outputTensors.size()))) {
          at::Tensor& input = inputTensors[r];
          at::Tensor& output = outputTensors[r];
          if (input.numel() != 0) {
            onecclSend(
                input.data_ptr(),
                input.numel(),
                getXcclDataType(input.scalar_type()),
                r,
                comm,
                &SyclQueue);
          }
          if (output.numel() != 0) {
            onecclRecv(
                output.data_ptr(),
                output.numel(),
                getXcclDataType(output.scalar_type()),
                r,
                comm,
                &SyclQueue);
          }
        }
        onecclGroupEnd();

        return;
      },
      OpType::ALLTOALL,
      "xccl:all_to_all");
}

} // namespace c10d

#endif // USE_C10D_XCCL
