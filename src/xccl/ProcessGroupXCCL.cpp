#ifdef USE_C10D_XCCL

#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <xccl/NanCheck_XPU.hpp>
#include <xccl/ProcessGroupXCCL.hpp>

namespace c10d {

using FlightRecorderXCCL = FlightRecorder<at::xpu::XPUEvent>;

namespace {

#if defined(CCL_MAJOR_VERSION) &&  \
    ((CCL_MAJOR_VERSION > 2021) || \
     (CCL_MAJOR_VERSION == 2021) && (CCL_MINOR_VERSION >= 15))
#define XCCL_HAS_AVG 1
#endif // oneCCL version >= 2021.15

const std::map<c10d::ReduceOp, ccl::reduction> xcclOps = {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
#ifdef XCCL_HAS_AVG
    {ReduceOp::AVG, ccl::reduction::avg},
#endif // XCCL_HAS_AVG
};

const std::map<at::ScalarType, ccl::datatype> xcclDatatypes = {
    {at::kByte, ccl::datatype::uint8},
    {at::kChar, ccl::datatype::int8},
    {at::kInt, ccl::datatype::int32},
    {at::kLong, ccl::datatype::int64},
    {at::kHalf, ccl::datatype::float16},
    {at::kFloat, ccl::datatype::float32},
    {at::kDouble, ccl::datatype::float64},
    {at::kBFloat16, ccl::datatype::bfloat16},
    {at::kBool, ccl::datatype::uint8},
    // use for non-reducetion op like allgather
    {at::kFloat8_e5m2, ccl::datatype::uint8},
    {at::kFloat8_e4m3fn, ccl::datatype::uint8},
    {at::kFloat8_e4m3fnuz, ccl::datatype::uint8},
    {at::kFloat8_e5m2fnuz, ccl::datatype::uint8},
};

bool computeLengthsAndCheckAndGetFlat(
    const std::vector<at::Tensor>& tensors,
    std::vector<size_t>& lengths,
    at::Tensor& flatTensor,
    int64_t& flatLength) {
  int64_t groupSize = tensors.size();
  auto firstTensor = tensors[0];
  int64_t totalSize = 0;
  bool isFlat = true;

  auto storage = firstTensor.storage();
  int64_t firstStorageOffset = firstTensor.storage_offset();

  for (int i = 0; i < groupSize; i++) {
    auto& curTensor = tensors[i];
    int64_t length = curTensor.numel();
    lengths[i] = length;
    totalSize += length;

    if (isFlat &&
        (!storage.is_alias_of(curTensor.storage()) ||
         curTensor.storage_offset() !=
             firstStorageOffset + totalSize - length)) {
      isFlat = false;
    }
  }

  flatLength = totalSize;

  if (isFlat) {
    flatTensor = firstTensor;
  } else {
    flatTensor = at::empty({totalSize}, firstTensor.options());
  }

  return isFlat;
}

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
  }

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

ccl::datatype getXcclDataType(
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

ccl::reduction getXcclReduceOp(const ReduceOp& reduceOp, at::Tensor& input) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        // Map sum to max for bool tensors to avoid overflow issues with sum.
        return ccl::reduction::max;
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
      return ccl::reduction::sum;
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

std::string dump_xccl_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive) {
  auto xcclDumpMap = std::unordered_map<
      std::string,
      std::unordered_map<std::string, std::string>>();
  return FlightRecorderXCCL::get()->dump(
      xcclDumpMap, includeCollectives, includeStackTraces, onlyActive);
}

constexpr int64_t kSynchronizeBusyWaitMillis = 10;
thread_local uint64_t ProcessGroupXCCL::xcclActiveGroupCounter_ = 0;

void TensorShelf::stash(std::vector<at::Tensor>& tensors) {
  std::lock_guard<std::mutex> lock(mutex_);
  tVector_.insert(tVector_.end(), tensors.begin(), tensors.end());
}

void TensorShelf::stash(TensorShelf& other) {
  std::vector<at::Tensor>& otherVec = other.get();
  this->stash(otherVec);
}

void TensorShelf::unstash() {
  this->clear();
}

bool TensorShelf::empty() {
  std::lock_guard<std::mutex> lock(mutex_);
  return tVector_.empty();
}

void TensorShelf::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  tVector_.clear();
}

std::vector<at::Tensor>& TensorShelf::get() {
  return tVector_;
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(
    at::Device& device,
    int rank,
    OpType opType,
    uint64_t seq,
    bool isP2P,
    const char* profilingTitle,
    const std::optional<std::vector<at::Tensor>>& inputs)
    : Work(rank, opType, profilingTitle, inputs),
      device_(device),
      workStartTime_(std::chrono::steady_clock::now()),
      seq_(seq),
      isP2P_(isP2P) {
  xcclEndEvent_ = std::make_shared<at::xpu::XPUEvent>();
  stashed_for_allocator_safety_ = std::make_shared<TensorShelf>();
}

ProcessGroupXCCL::WorkXCCL::WorkXCCL(const WorkXCCL& w)
    : Work(w.rank_, w.opType_),
      device_(w.device_),
      xcclEndEvent_(w.xcclEndEvent_),
      blockingWait_(w.blockingWait_),
      workStartTime_(w.workStartTime_),
      seq_(w.seq_),
      isP2P_(w.isP2P_),
      stashed_for_allocator_safety_(w.stashed_for_allocator_safety_) {}

ProcessGroupXCCL::WorkXCCL::~WorkXCCL() = default;

bool ProcessGroupXCCL::WorkXCCL::isCompleted() {
  if (xcclEndEvent_ && xcclEndEvent_->query()) {
    return true;
  }
  return false;
}

void ProcessGroupXCCL::WorkXCCL::synchronize() {
  synchronizeStream();
  if (c10d::allow_inflight_collective_as_graph_input()) {
    c10d::unregister_work(
        c10::intrusive_ptr<
            ProcessGroupXCCL::WorkXCCL>::unsafe_reclaim_from_nonowning(this));
  }
}

void ProcessGroupXCCL::WorkXCCL::synchronizeStream() {
  auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
  xcclEndEvent_->block(currentStream);
  stashed_for_allocator_safety_->unstash();
}

bool ProcessGroupXCCL::WorkXCCL::wait(std::chrono::milliseconds timeout) {
  synchronize();

  if (blockingWait_ || timeout != kNoTimeout) {
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
  } else if (isBarrierOp_ && !isCompleted()) {
    auto currentStream = at::xpu::getCurrentXPUStream(device_.index());
    currentStream.synchronize();
  }

  return true;
}

ProcessGroupXCCL::Options::Options(bool is_high_priority_stream)
    : Backend::Options(XCCL_BACKEND_NAME),
      is_high_priority_stream(is_high_priority_stream) {}

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
      store_(store),
      options_(std::move(options)),
      xcclCommCounter_(0),
      local_id_(process_group_id++) {
  logPrefix_ = createLogPrefix();
  blockingWait_ = getCvarBool(TORCH_XCCL_BLOCKING_WAIT, false);
  traceBufferSize_ = getCvarInt({"TORCH_FR_BUFFER_SIZE"}, 2000);

  this->setGroupUid(options_->group_name);
  // In PGNCCL, the pg ranks are recorded on comm setup in each op, but we just
  // do it here.
  const auto XcclVersion = getXcclVersion();
  FlightRecorderXCCL::get()->record_pg_ranks(
      std::make_tuple(pg_uid_, pg_desc_), groupRanks());
  FlightRecorderXCCL::get()->record_accelerator_version(XcclVersion);
  enableNanCheck_ = getCvarBool(TORCH_XCCL_NAN_CHECK, false);
  init();
  const std::string OFF = "OFF";
  std::string torch_distributed_debug =
      getCvarString({"TORCH_DISTRIBUTED_DEBUG"}, OFF.c_str());
  LOG(INFO) << logPrefix() << "ProcessGroupXCCL initialization options: "
            << "size: " << size << ", global rank: " << rank_
            << ", USE_HIGH_PRIORITY_STREAM: "
            << options_->is_high_priority_stream
            << ", PG Name: " << options_->group_name;

  LOG(INFO) << logPrefix() << "ProcessGroupXCCL environments: "
            << "XCCL version: " << XcclVersion
            << ", TORCH_XCCL_BLOCKING_WAIT: " << blockingWait_
            << ", TORCH_DISTRIBUTED_DEBUG: " << torch_distributed_debug
            << ", TORCH_XCCL_NAN_CHECK: " << enableNanCheck_;

  // Heartbeat monitor thread dumps debug info on write to pipe
  heartbeatMonitor_ = std::make_unique<HeartbeatMonitorXCCL>(this);
  heartbeatMonitor_->start();
}

ProcessGroupXCCL::~ProcessGroupXCCL() {
  heartbeatMonitor_->stop();
  // Wait for all threads to finish before returning
  heartbeatMonitor_->join();
}

bool ProcessGroupXCCL::dumpDebuggingInfo(bool includeStackTrace /*=true*/) {
  STATIC_SCOPED_WAIT_COUNTER(pytorch.ProcessGroupXCCL__dumpDebuggingInfo);
  LOG(ERROR)
      << logPrefix()
      << "ProcessGroupXCCL preparing to dump debug info. Include stack trace: "
      << includeStackTrace;
  if (traceBufferSize_ > 0) {
    // TODO: dump_xccl_trace
    auto xcclTrace = dump_xccl_trace(true, includeStackTrace, false);
    DebugInfoWriter& writer = DebugInfoWriter::getWriter(rank_);
    LOG(INFO) << logPrefix() << "ProcessGroupXCCL dumping xccl trace to "
              << writer.getWriterTarget();
    writer.write(xcclTrace);
    LOG(INFO) << logPrefix() << "Flight Recorder trace successfully dumped.";
    return true;
  }
  return false;
}

const std::vector<uint64_t>& ProcessGroupXCCL::groupRanks() const {
  if (options_->global_ranks_in_group.empty() && local_id_ == 0) {
    static std::vector<uint64_t> globalRanks(size_);
    std::iota(globalRanks.begin(), globalRanks.end(), 0);
    return globalRanks;
  }
  return options_->global_ranks_in_group;
}

void ProcessGroupXCCL::setEnqueuedPgStatus(
    c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work) {
  pgStatus_->lastEnqueuedSeq = static_cast<int64_t>(work->getSequencenumber());
  pgStatus_->lastEnqueuedWorkName = opTypeToString(work->opType_);
  pgStatus_->lastEnqueuedNumelIn = work->numelIn_;
  pgStatus_->lastEnqueuedNumelOut = work->numelOut_;
}

void ProcessGroupXCCL::setCompletedPgStatus(
    c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work) {
  pgStatus_->lastCompletedSeq = static_cast<int64_t>(work->getSequencenumber());
  pgStatus_->lastCompletedWorkName = opTypeToString(work->opType_);
  pgStatus_->lastCompletedNumelIn = work->numelIn_;
  pgStatus_->lastCompletedNumelOut = work->numelOut_;
  // To avoid complexity, we're not computing duration.
  FlightRecorderXCCL::get()->retire_id(
      work->trace_id_, /*compute_duration*/ false);
}

void ProcessGroupXCCL::setSequenceNumberForGroup() {}

uint64_t ProcessGroupXCCL::getSequenceNumberForGroup() {
  return seqCollective_;
}

void ProcessGroupXCCL::setEnableNanCheck(bool enableNanCheck) {
  enableNanCheck_ = enableNanCheck;
}

c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> ProcessGroupXCCL::initWork(
    at::Device& device,
    int rank,
    OpType opType,
    bool isP2P,
    const char* profilingTitle,
    const std::vector<at::Tensor>& inputs,
    const std::vector<at::Tensor>& outputs) {
  auto r = c10::make_intrusive<ProcessGroupXCCL::WorkXCCL>(
      device,
      rank,
      opType,
      isP2P ? seqP2P_ : seqCollective_,
      isP2P,
      profilingTitle,
      profilingTitle != nullptr ? std::optional<std::vector<at::Tensor>>(inputs)
                                : std::nullopt);

  r->trace_id_ = FlightRecorderXCCL::get()->record(
      local_id_,
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      seqCollective_,
      seqP2P_,
      op_id_,
      profilingTitle ? profilingTitle : "",
      inputs,
      outputs,
      nullptr,
      r->xcclEndEvent_.get(),
      options_->timeout,
      pgStatus_,
      isP2P);
  return r;
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

  bool batchP2P = xcclActiveGroupCounter_ > 0;
  bool singleP2POp = isP2POp(opType, batchP2P);

  c10::OptionalDeviceGuard gpuGuard(device);

  for (const auto i : c10::irange(xcclActiveGroupCounter_)) {
    (void)i;
    ccl::group_end();
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
  sycl::queue& q = c10::xpu::XPUStream(stream).queue();

  auto ctx = ccl::create_context(q.get_context());
  ccl::vector_class<ccl::pair_class<int, ccl::device>> devs_rank;
  devs_rank.emplace_back(rank, ccl::create_device(q.get_device()));

  auto xccl_kvs = get_kvs(rank_, *store_, singleP2POp, deviceKey, p2pRank);
  auto comms = ccl::create_communicators(numRanks, devs_rank, ctx, xccl_kvs);
  XCCLComm = std::make_shared<xcclComm_t>(std::move(comms[0]));

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
    ccl::group_start();
  }

  // The oneCCL group API requires retaining the SYCL queue (xcclstream) object
  // within the lifecycle of the communicator. If the XPU stream is created
  // within the collective operation, it would be destroyed earlier than the
  // communicator after the operation ends. Therefore, the XPU stream is stored
  // in a map alongside the communicator. Similarly, oneCCLv2 also requires
  // retaining the SYCL queue pointer for collective operations, so this change
  // will be necessary in oneCCLv2 as well.
  ccl::stream xccl_stream = ccl::create_stream(q);
  std::lock_guard<std::mutex> lock(mutex_);
  devXCCLCommMap_.emplace(deviceKey, XCCLComm);
  xcclStreamsMap_.emplace(
      deviceKey,
      std::make_pair(at::xpu::XPUStream(stream), std::move(xccl_stream)));
  xcclEventsMap_.emplace(deviceKey, at::xpu::XPUEvent());

  LOG(INFO) << logPrefix()
            << "Created XCCL communicator with Key: " << deviceKey;

  return XCCLComm;
}

void ProcessGroupXCCL::groupStart() {
  ccl::group_start();
  ++xcclActiveGroupCounter_;
}

void ProcessGroupXCCL::groupEnd() {
  ccl::group_end();
  --xcclActiveGroupCounter_;
}

static constexpr int CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;
void ProcessGroupXCCL::startCoalescing() {
  coalescedDevice_.set_index(-1);
  coalescedComm_ = nullptr;
  coalescedTensors_.clear();
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

  auto work = initWork(
      device,
      rank_,
      optype,
      coalescing_state_ & CoalP2P,
      "xccl:coalesced",
      {},
      {});

  work->blockingWait_ = blockingWait_;

  work->stashed_for_allocator_safety_->stash(coalescedTensors_);

  groupEnd();

  work->xcclEndEvent_->record(stream);
  setEnqueuedPgStatus(work);

  coalescing_state_ = 0;
  coalescedComm_ = nullptr;
  coalescedTensors_.clear();

  return coalescedAsync_ ? work : nullptr;
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
    bool asyncOp,
    const char* profilingTitle,
    bool nanCheck) {
  nanCheck &= enableNanCheck_;
  seqCollective_++;
  auto device = inputs[0].device();
  const auto key = std::to_string(device.index());
  auto comm = getXCCLComm(key, device, opType);

  if (coalescing_state_ & CoalActive) {
    if ((coalescing_state_ & CoalColl) == 0) {
      seqCollective_++;
    }
    op_id_++;
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
    coalescedAsync_ = asyncOp;
  }

  auto stream = asyncOp ? xcclStreamsMap_.at(key).first
                        : at::xpu::getCurrentXPUStream(device.index());

  std::unique_ptr<ccl::stream> cclstream;
  if (asyncOp) {
    cclstream = std::make_unique<ccl::stream>(xcclStreamsMap_.at(key).second);
    syncStream(device, xcclEventsMap_[key], stream);
  } else {
    auto StreamKey = key + "_" +
        std::to_string(at::xpu::getCurrentXPUStream(device.index()).id());
    if (xcclStreamsMap_.find(StreamKey) != xcclStreamsMap_.end()) {
      cclstream =
          std::make_unique<ccl::stream>(xcclStreamsMap_.at(StreamKey).second);
    } else {
      LOG(INFO) << "Current stream id changed, create new ccl stream";
      // update xcclStreamsMap_ with current stream key
      cclstream =
          std::make_unique<ccl::stream>(ccl::create_stream(stream.queue()));
      std::lock_guard<std::mutex> lock(mutex_);
      xcclStreamsMap_.emplace(
          StreamKey, std::make_pair(at::xpu::XPUStream(stream), *cclstream));
    }
  }

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
  work =
      initWork(device, rank_, opType, false, profilingTitle, inputs, outputs);
  if (coalescing_state_) {
    FlightRecorderXCCL::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle ? profilingTitle : "",
        inputs,
        outputs,
        nullptr,
        nullptr,
        options_->timeout,
        pgStatus_,
        false);
  }

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  if (asyncOp) {
    if (coalescing_state_) {
      coalescedTensors_.stash(inputs);
      coalescedTensors_.stash(outputs);
    } else {
      work->stashed_for_allocator_safety_->stash(inputs);
      work->stashed_for_allocator_safety_->stash(outputs);
    }
  }

  c10::OptionalDeviceGuard gpuGuard(device);

  if (nanCheck) {
    for (const auto& input : inputs) {
      checkForNan(input, stream);
    }
  }

  pre(stream, work);

  for (const auto i : c10::irange(inputs.size())) {
    fn(inputs[i], outputs[i], *comm, stream, *cclstream);
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
  work->future_->addCallback(
      [this, work](at::ivalue::Future&) { this->setCompletedPgStatus(work); });
  work->blockingWait_ = blockingWait_;

  work->numelIn_ = 0;
  work->numelOut_ = 0;
  for (const auto& input : inputs) {
    work->numelIn_ += input.numel();
  }
  for (const auto& output : outputs) {
    work->numelOut_ += output.numel();
  }
  setEnqueuedPgStatus(work);

  return asyncOp ? work : nullptr;
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

  op_id_++;
  auto comm = getXCCLComm(key, device, opType, p2pRank, isSendRecvSelf);

  if (coalescing_state_ & CoalActive) {
    if ((coalescing_state_ & CoalP2P) == 0) {
      seqP2P_++;
    }
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
    coalescedAsync_ = true;
  }

  auto stream = xcclStreamsMap_.at(key).first;
  auto cclstream = xcclStreamsMap_.at(key).second;
  syncStream(device, xcclEventsMap_[key], stream);

  if (enableNanCheck_ && opType == OpType::SEND) {
    checkForNan(tensor, stream);
  }

  if (!coalescing_state_) {
    auto work =
        initWork(device, rank_, opType, true, profilingTitle, {tensor}, {});
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>();
    work->outputs_->push_back(tensor);

    work->trace_id_ = FlightRecorderXCCL::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle,
        {tensor},
        {tensor},
        nullptr,
        work->xcclEndEvent_.get(),
        options_->timeout,
        pgStatus_,
        true);

    c10::OptionalDeviceGuard gpuGuard(device);

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
    work->future_->addCallback([this, work](at::ivalue::Future&) {
      this->setCompletedPgStatus(work);
    });

    work->numelIn_ = work->numelOut_ = tensor.numel();
    setEnqueuedPgStatus(work);
    return work;
  } else {
    FlightRecorderXCCL::get()->record(
        local_id_,
        std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
        seqCollective_,
        seqP2P_,
        op_id_,
        profilingTitle,
        {tensor},
        {tensor},
        nullptr,
        nullptr,
        options_->timeout,
        pgStatus_,
        true);
    c10::OptionalDeviceGuard gpuGuard(device);

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
      this->getSize(), // worldSize
      "N/A", // async_op
      "N/A"); // reductionOp

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& input,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream,
          int dst) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::send(
            input.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            dst,
            comm,
            xcclStream);
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
      this->getSize(), // worldSize
      "N/A", // async_op
      "N/A"); // reductionOp

  auto ret = pointToPoint(
      tensor,
      [&](at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream,
          int src) {
        auto xcclDataType = getXcclDataType(output.scalar_type());
        ccl::recv(
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            src,
            comm,
            xcclStream);
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
  checkSingleTensor(inputTensor);
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      "N/A"); // reductionOp

  auto inputs = std::vector<at::Tensor>{inputTensor};
  return collective(
      inputs,
      outputs, // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
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
                ccl::recv(
                    outputs[r].data_ptr(),
                    (size_t)inputTensor.numel(),
                    xcclDataType,
                    r,
                    comm,
                    xcclStream);
              } else {
                // on its own rank, simply copy from the input
                outputs[r].copy_(inputTensor);
              }
            }
          } else {
            // do send
            ccl::send(
                inputTensor.data_ptr(),
                (size_t)inputTensor.numel(),
                xcclDataType,
                root,
                comm,
                xcclStream);
          }
          return;
        }
      },
      OpType::GATHER,
      opts.asyncOp,
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
    // if not in the root rank, initialize inputTensors as empty place holder
    // with an empty list
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      "N/A"); // reductionOp

  const auto root = opts.rootRank;
  bool nanCheck = (rank_ == root);

  auto outputs = std::vector<at::Tensor>{outputTensor};
  return collective(
      outputs,
      inputs, // just to fit the collective interface
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
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
                ccl::send(
                    inputs[r].data_ptr(),
                    send_count,
                    send_type,
                    r,
                    comm,
                    xcclStream);
              } else {
                // on its own rank, simply copy from the input
                outputTensor.copy_(inputs[r]);
              }
            }
          } else {
            // do receive
            size_t recv_count = outputTensor.numel();
            auto recv_type = getXcclDataType(outputTensor.scalar_type());
            ccl::recv(
                outputTensor.data_ptr(),
                recv_count,
                recv_type,
                root,
                comm,
                xcclStream);
          }

          return;
        }
      },
      OpType::SCATTER,
      opts.asyncOp,
      "xccl:scatter",
      nanCheck);
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
          ccl::stream& xcclStream) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            xcclStream);
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
      opts.asyncOp,
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
      size_, // worldSize
      opts.asyncOp, // async_op
      reduceOpToString(opts.reduceOp)); // reductionOp

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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      reduceOpToString(opts.reduceOp)); // reductionOp

  return collectiveCoalesced(
      tensors,
      tensors,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            xcclStream);
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
      opts.asyncOp,
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      "N/A"); // reductionOp

  const auto root = opts.rootRank + opts.rootTensor;
  bool nanCheck = (root == rank_);

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::broadcast(
            input.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            root,
            comm,
            xcclStream);
        return;
      },
      OpType::BROADCAST,
      opts.asyncOp,
      "xccl:broadcast",
      nanCheck);
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
  bool nanCheck = (root == rank_);
  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::broadcast(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            root,
            comm,
            xcclStream);
        return;
      },
      OpType::BROADCAST,
      opts.asyncOp,
      "xccl:_broadcast_oop",
      nanCheck);
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      reduceOpToString(opts.reduceOp)); // reductionOp

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
        const int root = opts.rootRank + opts.rootTensor;
        const auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        const auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::reduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            root,
            comm,
            xcclStream);
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
      opts.asyncOp,
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
          ccl::stream& xcclStream) {
        const int root = opts.rootRank + opts.rootTensor;
        const auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        const auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::reduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            root,
            comm,
            xcclStream);
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
      opts.asyncOp,
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      "N/A"); // reductionOp

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
            ccl::stream& xcclStream) {
          auto xcclDataType = getXcclDataType(input.scalar_type());
          ccl::allgather(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)input.numel(),
              xcclDataType,
              comm,
              xcclStream);
          return;
        },
        [](at::xpu::XPUStream&,
           c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {},
        [&](at::xpu::XPUStream& Stream,
            c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>& work) {
          if (opts.asyncOp) {
            work->stashed_for_allocator_safety_->stash(outputTensors_);
          }
          // Copy the flattened output tensors to the outputs.
          c10::StreamGuard guard(Stream);
          for (const auto j : c10::irange(outputTensors_.size())) {
            outputTensors_[j].copy_(outputFlattened[j], true);
          }
        },
        OpType::ALLGATHER,
        opts.asyncOp,
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      "N/A"); // reductionOp

  return collective(
      input_tensor,
      output_tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::allgather(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            comm,
            xcclStream);
        return;
      },
      OpType::_ALLGATHER_BASE,
      opts.asyncOp,
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      "N/A"); // reductionOp

  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::allgather(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            comm,
            xcclStream);
        return;
      },
      OpType::COALESCED,
      opts.asyncOp,
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      reduceOpToString(opts.reduceOp)); // reductionOp

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
            ccl::stream& xcclStream) {
          auto xcclDataType = getXcclDataType(input.scalar_type(), true);
          auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
          ccl::reduce_scatter(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)output.numel(),
              xcclDataType,
              xcclReduceOp,
              comm,
              xcclStream);
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
          if (opts.asyncOp) {
            work->stashed_for_allocator_safety_->stash(inputTensors_);
          }
          // Copy the input tensors to the flattened inputs.
          c10::StreamGuard guard(Stream);
          for (const auto j : c10::irange(inputTensors_.size())) {
            inputFlattened[j].copy_(inputTensors_[j], true);
          }
        },
        [&](at::xpu::XPUStream&,
            c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL>&) {},
        OpType::REDUCE_SCATTER,
        opts.asyncOp,
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      reduceOpToString(opts.reduceOp)); // reductionOp

  return collective(
      inputTensor,
      outputTensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::reduce_scatter(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            xcclStream);
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
      opts.asyncOp,
      "xccl:_reduce_scatter_base");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const ReduceScatterOptions& opts) {
  RECORD_PARAM_COMMS_DATA_WITH_LOG(
      std::make_tuple(
          static_cast<int64_t>(seqCollective_) + 1,
          false), // seq + 1 to match collective and assume only one collective
                  // in coalesced range
      std::make_tuple(pg_uid_, pg_desc_), // PG name tuple
      inputs, // inputTensors
      outputs, // outputTensors
      rank_, // rank
      "reduce_scatter_tensor_coalesced", // collective name
      getTensorsNumel(inputs), // inNelems
      getTensorsNumel(outputs), // outNelems
      inputs[0].scalar_type(), // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // globalRankStart
      -1, // globalRankStride
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      reduceOpToString(opts.reduceOp)); // reductionOp

  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::reduce_scatter(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            xcclStream);
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
      opts.asyncOp,
      "xccl:reduce_scatter_tensor_coalesced");
}

c10::DeviceIndex ProcessGroupXCCL::guessDeviceId() const {
  if (getBoundDeviceId().has_value()) {
    return getBoundDeviceId().value().index();
  } else if (!usedDeviceIdxs_.empty()) {
    return *usedDeviceIdxs_.begin();
  }
  int devIdx =
      static_cast<int16_t>(rank_ % at::detail::getXPUHooks().getNumGPUs());
  LOG(WARNING)
      << logPrefix()
      << c10::str(
             " using GPU ",
             devIdx,
             " as device used by this process is currently unknown. ",
             "This can potentially cause a hang if this rank to GPU mapping is incorrect. ",
             "You can specify device_id in init_process_group() to force use of a particular device.");
  return static_cast<c10::DeviceIndex>(devIdx);
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
  c10::DeviceIndex barDevIdx = -1;

  // See nccl barrier comments
  if (!opts.device_ids.empty()) {
    barDevIdx = static_cast<c10::DeviceIndex>(opts.device_ids[0]);
  } else {
    barDevIdx = guessDeviceId();
  }

  TORCH_CHECK_WITH(
      ValueError,
      barDevIdx >= 0,
      "Failed to infer a GPU device id to perform barrier. ");
  auto barDevice = at::Device(at::DeviceType::XPU, barDevIdx);

  at::Tensor barrierTensor =
      at::zeros({1}, at::TensorOptions().device(barDevice).dtype(at::kFloat));

  AllreduceOptions arOpts = AllreduceOptions();
  arOpts.asyncOp = opts.asyncOp;
  auto work = allreduce_impl(barrierTensor, "xccl:all_reduce_barrier", arOpts);

  if (opts.asyncOp) {
    auto xcclWork = dynamic_cast<ProcessGroupXCCL::WorkXCCL*>(work.get());
    TORCH_CHECK(xcclWork);
    xcclWork->isBarrierOp_ = true;
    return work;
  }

  auto currentStream = at::xpu::getCurrentXPUStream(barDevIdx);
  currentStream.synchronize();
  return nullptr;
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts) {
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
        this->getSize(), // worldSize
        opts.asyncOp, // async_op
        "N/A"); // reductionOp

    TORCH_CHECK(
        outputTensor.numel() == inputTensor.numel() &&
            outputTensor.scalar_type() == inputTensor.scalar_type(),
        "xpu_alltoall_base: tensors are not equal in size or data type");
    TORCH_CHECK(
        outputTensor.size(0) % size_ == 0,
        "xpu_alltoall_base: tensor's dim 0 does not divide equally across group size");
    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            xcclComm_t& comm,
            at::xpu::XPUStream& stream,
            ccl::stream& xcclStream) {
          auto xcclDataType = getXcclDataType(output.scalar_type());
          ccl::alltoall(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)output.numel() / comm.size(),
              xcclDataType,
              comm,
              xcclStream);
          return;
        },
        OpType::ALLTOALL_BASE,
        opts.asyncOp,
        "xccl:all_to_all");
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
        this->getSize(), // worldSize
        opts.asyncOp, // async_op
        "N/A"); // reductionOp

    return collective(
        inputTensor,
        outputTensor,
        [&](at::Tensor& input,
            at::Tensor& output,
            xcclComm_t& comm,
            at::xpu::XPUStream& stream,
            ccl::stream& xcclStream) {
          std::vector<size_t> sendCounts(size_);
          std::vector<size_t> recvCounts(size_);
          bool inputSplitsEqual = inputSplitSizes.size() == 0;
          bool outputSplitsEqual = outputSplitSizes.size() == 0;

          size_t inLen = input.numel();
          size_t outLen = output.numel();
          if (inLen)
            inLen /= (inputSplitsEqual ? size_ : input.size(0));
          if (outLen)
            outLen /= (outputSplitsEqual ? size_ : output.size(0));

          for (int i = 0; i < size_; i++) {
            sendCounts[i] =
                (inputSplitsEqual ? inLen : inputSplitSizes[i] * inLen);
            recvCounts[i] =
                (outputSplitsEqual ? outLen : outputSplitSizes[i] * outLen);
          }
          auto xcclDataType = getXcclDataType(output.scalar_type());
          ccl::alltoallv(
              input.data_ptr(),
              sendCounts,
              output.data_ptr(),
              recvCounts,
              xcclDataType,
              comm,
              xcclStream);
          return;
        },
        OpType::ALLTOALL_BASE,
        opts.asyncOp,
        "xccl:all_to_all");
  }
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
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
      this->getSize(), // worldSize
      opts.asyncOp, // async_op
      "N/A"); // reductionOp

  return collective(
      inputTensors,
      outputTensors,
      [&](at::Tensor& /* unused */,
          at::Tensor& /* unused */,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream,
          ccl::stream& xcclStream) {
        c10::OptionalStreamGuard stream_guard(stream.unwrap());
        at::Tensor flatInput;
        at::Tensor flatOutput;

        std::vector<size_t> sendCounts(size_);
        std::vector<size_t> recvCounts(size_);

        int64_t flatSendCount;
        int64_t flatRecvCount;

        bool isInputFlat = computeLengthsAndCheckAndGetFlat(
            inputTensors, sendCounts, flatInput, flatSendCount);
        bool isOutputFlat = computeLengthsAndCheckAndGetFlat(
            outputTensors, recvCounts, flatOutput, flatRecvCount);
        if (!isInputFlat) {
          auto flatInputSplits = flatInput.split_with_sizes(
              c10::IntArrayRef((int64_t*)sendCounts.data(), sendCounts.size()),
              0);

          for (int i = 0; i < size_; i++) {
            flatInputSplits[i].copy_(inputTensors[i].view({-1}));
          }
        }

        auto xcclDataType = getXcclDataType(flatOutput.scalar_type());
        ccl::event ret_evt;
        ret_evt = ccl::alltoallv(
            flatInput.data_ptr(),
            sendCounts,
            flatOutput.data_ptr(),
            recvCounts,
            xcclDataType,
            comm,
            xcclStream);

        if (!isOutputFlat) {
          ret_evt.wait();
          auto flatOutputSplits = flatOutput.split_with_sizes(
              c10::IntArrayRef((int64_t*)recvCounts.data(), recvCounts.size()),
              0);

          for (int i = 0; i < size_; i++) {
            outputTensors[i].view({-1}).copy_(flatOutputSplits[i]);
          }
        }
        stream.synchronize();
        return;
      },
      OpType::ALLTOALL,
      opts.asyncOp,
      "xccl:all_to_all");
}

std::string getXcclVersion() {
  auto xccl_version = ccl::get_library_version();
  std::string versionString = std::to_string(xccl_version.major) + "." +
      std::to_string(xccl_version.minor) + "." +
      std::to_string(xccl_version.update);
  return versionString;
}

} // namespace c10d

#endif // USE_C10D_XCCL
