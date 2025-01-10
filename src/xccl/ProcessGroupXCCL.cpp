#ifdef USE_C10D_XCCL

#include <comm/XPUGuard.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>
#include <xccl/ProcessGroupXCCL.hpp>

namespace c10d {

namespace {
const std::map<c10d::ReduceOp, ccl::reduction> xcclOps = {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
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
  if (!tensor.is_xpu() || tensor.is_sparse() || tensor.is_complex()) {
    C10_THROW_ERROR(
        ValueError, "Tensors must be XPU and dense and non-complex");

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
    if (!t.is_xpu() || t.is_sparse() || t.is_complex()) {
      C10_THROW_ERROR(
          ValueError, "Tensors must be XPU and dense and non-complex");
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
    if (input.scalar_type() == at::kBool && reduceOp == ReduceOp::SUM) {
      // Map sum to max for bool tensors to avoid overflow issues with sum.
      return ccl::reduction::max;
    }
    // Use SUM emu AVG due to oneCCL not support AVG.
    // oneCCL is expected to support avg in basekit 2025.2 release.
    if (reduceOp == ReduceOp::AVG) {
      return ccl::reduction::sum;
    }
    return xcclOps.at(reduceOp);
  } catch (const std::out_of_range&) {
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduceOpToString(reduceOp) + " with XCCL");
  }
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

constexpr const char* MULTI_DEVICE_ERROR_MSG =
    "Expecting one tensor only but got multiple";

ProcessGroupXCCL::ProcessGroupXCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size)
    : Backend(rank, size), store_(store), xcclCommCounter_(0) {
  blockingWait_ = getCvarBool(TORCH_XCCL_BLOCKING_WAIT, false);
  init();
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

  at::xpu::OptionalXPUGuard gpuGuard(device);

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

  c10::impl::VirtualGuardImpl impl(device.type());
  c10::Stream stream =
      impl.getStreamFromGlobalPool(device, /*isHighPriority=*/false);
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

  std::lock_guard<std::mutex> lock(mutex_);
  devXCCLCommMap_.emplace(deviceKey, XCCLComm);
  xcclStreamsMap_.emplace(deviceKey, std::move(stream));
  xcclEventsMap_.emplace(deviceKey, at::xpu::XPUEvent());

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
  auto stream = xcclStreamsMap_.at(key);

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

  auto stream = xcclStreamsMap_.at(key);
  syncStream(device, xcclEventsMap_[key], stream);

  c10::intrusive_ptr<ProcessGroupXCCL::WorkXCCL> work;
  work = initWork(device, rank_, opType);

  work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

  at::xpu::OptionalXPUGuard gpuGuard(device);

  pre(stream, work);

  for (const auto i : c10::irange(inputs.size())) {
    c10::xpu::XPUCachingAllocator::recordStream(
        inputs[i].storage().data_ptr(), stream);
    fn(inputs[i], outputs[i], *comm, stream);
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

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce_impl(
    at::Tensor& tensor,
    const AllreduceOptions& opts) {
  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        auto ccl_stream = ccl::create_stream(stream.queue());
        ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl::create_stream(stream.queue()));
        // Use SUM emu AVG due to oneCCL not support AVG
        // oneCCL is expected to support avg in basekit 2025.2 release.
        if (opts.reduceOp == ReduceOp::AVG) {
          auto divisor = getSize();
          output.div_(divisor);
        }
        return;
      },
      OpType::ALLREDUCE,
      "xccl:all_reduce");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, MULTI_DEVICE_ERROR_MSG);
  auto tensor = tensors.back();
  checkSingleTensor(tensor);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
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

  return collective(
      tensor,
      tensor,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl::create_stream(stream.queue()));
        // Use SUM emu AVG due to oneCCL not support AVG
        // oneCCL is expected to support avg in basekit 2025.2 release.
        if (opts.reduceOp == ReduceOp::AVG) {
          auto divisor = getSize();
          output.div_(divisor);
        }
        return;
      },
      OpType::ALLREDUCE,
      "xccl:all_reduce");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  auto total_numel = checkTensorOnSameDevice(tensors);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
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
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::allreduce(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl::create_stream(stream.queue()));
        // Use SUM emu AVG due to oneCCL not support AVG
        // oneCCL is expected to support avg in basekit 2025.2 release.
        if (opts.reduceOp == ReduceOp::AVG) {
          auto divisor = getSize();
          output.div_(divisor);
        }
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
  checkSingleTensor(tensor);

  // @lint-ignore CLANGTIDY
  RECORD_PARAM_COMMS_DATA(
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
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::broadcast(
            input.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            root,
            comm,
            ccl::create_stream(stream.queue()));
        return;
      },
      OpType::BROADCAST,
      "nccl:broadcast");
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
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::broadcast(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            root,
            comm,
            ccl::create_stream(stream.queue()));
        return;
      },
      OpType::BROADCAST,
      "xccl:_broadcast_oop");
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
          at::xpu::XPUStream& stream) {
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
            ccl::create_stream(stream.queue()));
        // Use SUM emu AVG due to oneCCL not support AVG
        // oneCCL is expected to support avg in basekit 2025.2 release.
        if (opts.reduceOp == ReduceOp::AVG && getRank() == root) {
          auto divisor = getSize();
          output.div_(divisor);
        }
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

  RECORD_PARAM_COMMS_DATA(
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
            at::xpu::XPUStream& stream) {
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          auto xcclDataType = getXcclDataType(input.scalar_type());
          ccl::allgather(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)input.numel(),
              xcclDataType,
              comm,
              ccl::create_stream(stream.queue()));
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

  RECORD_PARAM_COMMS_DATA(
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
          at::xpu::XPUStream& stream) {
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::allgather(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            comm,
            ccl::create_stream(stream.queue()));
        return;
      },
      OpType::_ALLGATHER_BASE,
      "xccl:_all_gather_base");
}

c10::intrusive_ptr<Work> ProcessGroupXCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs,
    std::vector<at::Tensor>& inputs,
    const AllgatherOptions& opts) {
  return collectiveCoalesced(
      inputs,
      outputs,
      [&](at::Tensor& input,
          at::Tensor& output,
          xcclComm_t& comm,
          at::xpu::XPUStream& stream) {
        auto xcclDataType = getXcclDataType(input.scalar_type());
        ccl::allgather(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)input.numel(),
            xcclDataType,
            comm,
            ccl::create_stream(stream.queue()));
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

  RECORD_PARAM_COMMS_DATA(
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
            at::xpu::XPUStream& stream) {
          c10::xpu::XPUCachingAllocator::recordStream(
              output.storage().data_ptr(), stream);
          auto xcclDataType = getXcclDataType(input.scalar_type(), true);
          auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
          ccl::reduce_scatter(
              input.data_ptr(),
              output.data_ptr(),
              (size_t)output.numel(),
              xcclDataType,
              xcclReduceOp,
              comm,
              ccl::create_stream(stream.queue()));
          // Use SUM emu AVG due to oneCCL not support AVG
          // oneCCL is expected to support avg in basekit 2025.2 release.
          if (opts.reduceOp == ReduceOp::AVG) {
            auto divisor = getSize();
            output.div_(divisor);
          }
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

  RECORD_PARAM_COMMS_DATA(
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
          at::xpu::XPUStream& stream) {
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::reduce_scatter(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl::create_stream(stream.queue()));
        // Use SUM emu AVG due to oneCCL not support AVG
        // oneCCL is expected to support avg in basekit 2025.2 release.
        if (opts.reduceOp == ReduceOp::AVG) {
          auto divisor = getSize();
          output.div_(divisor);
        }
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
          at::xpu::XPUStream& stream) {
        c10::xpu::XPUCachingAllocator::recordStream(
            output.storage().data_ptr(), stream);
        auto xcclDataType = getXcclDataType(input.scalar_type(), true);
        auto xcclReduceOp = getXcclReduceOp(opts.reduceOp, input);
        ccl::reduce_scatter(
            input.data_ptr(),
            output.data_ptr(),
            (size_t)output.numel(),
            xcclDataType,
            xcclReduceOp,
            comm,
            ccl::create_stream(stream.queue()));
        // Use SUM emu AVG due to oneCCL not support AVG
        // oneCCL is expected to support avg in basekit 2025.2 release.
        if (opts.reduceOp == ReduceOp::AVG) {
          auto divisor = getSize();
          output.div_(divisor);
        }
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

  auto work = allreduce_impl(barrierTensor);

  auto xcclWork = dynamic_cast<ProcessGroupXCCL::WorkXCCL*>(work.get());
  TORCH_CHECK(xcclWork);
  xcclWork->barrierTensor_ = std::move(barrierTensor);
  return work;
}

} // namespace c10d

#endif // USE_C10D_XCCL
