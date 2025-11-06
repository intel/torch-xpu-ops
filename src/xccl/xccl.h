#pragma once

#define CCL_ENABLE_ZE
#define CCL_ENABLE_SYCL

#include <ATen/xpu/XPUEvent.h>
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <oneapi/ccl.h>
#include <oneapi/ccl.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#if defined(CCL_MAJOR_VERSION) &&  \
    ((CCL_MAJOR_VERSION > 2021) || \
     (CCL_MAJOR_VERSION == 2021) && (CCL_MINOR_VERSION >= 15))
#define XCCL_HAS_AVG 1
#endif // oneCCL version >= 2021.15

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

inline bool isCCLV2EnabledCached() {
  static const bool cachedValue = []() {
    const char* use_ccl_v2_env = std::getenv("USE_CCL_V2");
    if (use_ccl_v2_env) {
      std::string value(use_ccl_v2_env);
      std::transform(
          value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return std::tolower(c);
          });
      return value == "on" || value == "yes" || value == "1";
    }
    return false;
  }();
  return cachedValue;
}

inline const std::string& getVersionString() {
static std::string versionString = []() {

...
return versionString;
  bool useCCLV2 = isCCLV2EnabledCached();
  std::string versionString;
  if (useCCLV2) {
    int version = 0;
    onecclGetVersion(&version);
    const int majorBase = 10000;
    const int minorBase = 100;
    auto xcclMajor = version / majorBase;
    auto xcclMinor = (version % majorBase) / minorBase;
    auto xcclPatch = version % (xcclMajor * majorBase + xcclMinor * minorBase);
    versionString = std::to_string(xcclMajor) + "." +
        std::to_string(xcclMinor) + "." + std::to_string(xcclPatch);
  } else {
    auto xccl_version = ccl::get_library_version();
    versionString = std::to_string(xccl_version.major) + "." +
        std::to_string(xccl_version.minor) + "." +
        std::to_string(xccl_version.update);
  }
  return versionString;
}();

namespace c10d {

struct XCCLStream {
  at::xpu::XPUStream xpuStream;
  ccl::stream cclStream;
};

struct xcclComm_t {
  std::optional<ccl::communicator> cclComm;
  onecclComm_t onecclComm{nullptr};

  xcclComm_t() = default;
  explicit xcclComm_t(ccl::communicator comm)
      : cclComm(std::move(comm)), onecclComm(nullptr) {}
  explicit xcclComm_t(onecclComm_t comm) : onecclComm(comm) {}
};

const std::map<c10d::ReduceOp, onecclRedOp_t> xcclOpsV2 = {
    {ReduceOp::MIN, onecclRedOp_t::onecclMin},
    {ReduceOp::MAX, onecclRedOp_t::onecclMax},
    {ReduceOp::SUM, onecclRedOp_t::onecclSum},
    {ReduceOp::PRODUCT, onecclRedOp_t::onecclProd},
#ifdef XCCL_HAS_AVG
    {ReduceOp::AVG, onecclRedOp_t::onecclAvg},
#endif // XCCL_HAS_AVG
};

const std::map<c10d::ReduceOp, ccl::reduction> xcclOpsV1 = {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
#ifdef XCCL_HAS_AVG
    {ReduceOp::AVG, ccl::reduction::avg},
#endif // XCCL_HAS_AVG
};

inline const std::map<at::ScalarType, onecclDataType_t> xcclDatatypesV2 = {
    {at::kByte, onecclDataType_t::onecclUint8},
    {at::kChar, onecclDataType_t::onecclChar},
    {at::kInt, onecclDataType_t::onecclInt32},
    {at::kLong, onecclDataType_t::onecclInt64},
    {at::kHalf, onecclDataType_t::onecclFloat16},
    {at::kFloat, onecclDataType_t::onecclFloat32},
    {at::kDouble, onecclDataType_t::onecclFloat64},
    {at::kBFloat16, onecclDataType_t::onecclBfloat16},
    {at::kBool, onecclDataType_t::onecclUint8},
    // use for non-reduction op like allgather
    {at::kFloat8_e5m2, onecclDataType_t::onecclUint8},
    {at::kFloat8_e4m3fn, onecclDataType_t::onecclUint8},
    {at::kFloat8_e4m3fnuz, onecclDataType_t::onecclUint8},
    {at::kFloat8_e5m2fnuz, onecclDataType_t::onecclUint8},
};

const std::map<at::ScalarType, ccl::datatype> xcclDatatypesV1 = {
    {at::kByte, ccl::datatype::uint8},
    {at::kChar, ccl::datatype::int8},
    {at::kInt, ccl::datatype::int32},
    {at::kLong, ccl::datatype::int64},
    {at::kHalf, ccl::datatype::float16},
    {at::kFloat, ccl::datatype::float32},
    {at::kDouble, ccl::datatype::float64},
    {at::kBFloat16, ccl::datatype::bfloat16},
    {at::kBool, ccl::datatype::uint8},
    // use for non-reduction op like allgather
    {at::kFloat8_e5m2, ccl::datatype::uint8},
    {at::kFloat8_e4m3fn, ccl::datatype::uint8},
    {at::kFloat8_e4m3fnuz, ccl::datatype::uint8},
    {at::kFloat8_e5m2fnuz, ccl::datatype::uint8},
};

namespace {

ccl::datatype getXcclDataTypeV1(
    at::ScalarType type,
    bool is_reduction_op = false) {
  if (is_reduction_op) {
    TORCH_CHECK(
        !isFloat8Type(type),
        "Float8 dtypes are not currently supported for XCCL reductions");
  }
  auto it = xcclDatatypesV1.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      it != xcclDatatypesV1.end(),
      "Input tensor data type is not supported for XCCL process group: ",
      type);
  return it->second;
}

// V2 specific function to avoid variant overhead
onecclDataType_t getXcclDataTypeV2(
    at::ScalarType type,
    bool is_reduction_op = false) {
  if (is_reduction_op) {
    TORCH_CHECK(
        !isFloat8Type(type),
        "Float8 dtypes are not currently supported for XCCL reductions");
  }
  auto it = xcclDatatypesV2.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      it != xcclDatatypesV2.end(),
      "Input tensor data type is not supported for XCCL process group: ",
      type);
  return it->second;
}

ccl::reduction getXcclReduceOpV1(const ReduceOp& reduceOp, at::Tensor& input) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
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
      LOG(INFO) << "[Reduce] Use sum emulation for avg";
      return ccl::reduction::sum;
    }
#endif
    return xcclOpsV1.at(reduceOp);
  } catch (const std::out_of_range&) {
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduceOpToString(reduceOp) + " with XCCL");
  }
}

onecclRedOp_t getXcclReduceOpV2(const ReduceOp& reduceOp, at::Tensor& input) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
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
      LOG(INFO) << "[Reduce] Use sum emulation for avg";
      return onecclRedOp_t::onecclSum;
    }
#endif
    return xcclOpsV2.at(reduceOp);
  } catch (const std::out_of_range&) {
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduceOpToString(reduceOp) + " with XCCL");
  }
}

std::mutex kvs_mutex;

ccl::shared_ptr_class<ccl::kvs> get_kvs(
    int rank,
    Store& store,
    uint64_t& xcclCommCounter_,
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

void broadcastUniqueXCCLID(
    onecclUniqueId* xcclID,
    bool isSingleP2POp,
    const std::string& p2pKey,
    int p2pRank,
    uint64_t& xcclCommCounter_,
    int rank_,
    Store* store_) {
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

std::shared_ptr<xcclComm_t> createXCCLCommHelper(
    int rank,
    int numRanks,
    int rank_,
    c10::Device& device,
    sycl::queue& q,
    const std::string& deviceKey,
    Store* store_,
    bool singleP2POp,
    int p2pRank,
    uint64_t& xcclCommCounter_) {
  bool useCCLV2 = isCCLV2EnabledCached();
  if (!useCCLV2) {
    auto ctx = ccl::create_context(q.get_context());
    ccl::vector_class<ccl::pair_class<int, ccl::device>> devs_rank;
    devs_rank.emplace_back(rank, ccl::create_device(q.get_device()));

    auto xccl_kvs = get_kvs(
        rank_, *store_, xcclCommCounter_, singleP2POp, deviceKey, p2pRank);
    auto comms = ccl::create_communicators(numRanks, devs_rank, ctx, xccl_kvs);
    return std::make_shared<xcclComm_t>(std::move(comms[0]));
  } else {
    LOG(INFO) << "USE_CCL_V2=1";
    onecclUniqueId xcclID;
    if (rank_ == 0 || (singleP2POp && p2pRank == 0)) {
      onecclGetUniqueId(&xcclID);
    }
    broadcastUniqueXCCLID(
        &xcclID,
        singleP2POp,
        deviceKey,
        p2pRank,
        xcclCommCounter_,
        rank_,
        store_);
    onecclComm_t comm = nullptr;
    onecclResult_t result = onecclSuccess;
    result = onecclSetDevice(device.index());
    if (result != onecclSuccess) {
      std::cerr << "Failed to set device.\n";
    }
    result = onecclCommInitRank(&comm, numRanks, xcclID, rank);
    if (result != onecclSuccess) {
      std::cerr << "Failed to initialize communicator.\n";
    }
    return std::make_shared<xcclComm_t>(comm);
  }
}

} // namespace

namespace xccl {

void oneccl_group_start();
void oneccl_group_end();
void onecclAllReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    ccl::stream& xcclStream,
    void* SyclQueue);
void onecclReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    const int root,
    ccl::stream& xcclStream,
    void* SyclQueue);
void onecclBroadcast(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const int root,
    ccl::stream& xcclStream,
    void* SyclQueue);
void onecclReduceScatter(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    ccl::stream& xcclStream,
    void* SyclQueue);
void onecclAllGather(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    ccl::stream& xcclStream,
    void* SyclQueue);
void onecclSend(
    at::Tensor& input,
    xcclComm_t& comm,
    const int dstRank,
    ccl::stream& xcclStream,
    void* SyclQueue);
void onecclRecv(
    at::Tensor& output,
    xcclComm_t& comm,
    const int srcRank,
    ccl::stream& xcclStream,
    void* SyclQueue);
void onecclGather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    xcclComm_t& comm,
    const int root,
    ccl::stream& xcclStream,
    void* SyclQueue);
void onecclScatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    xcclComm_t& comm,
    const int root,
    ccl::stream& xcclStream,
    void* SyclQueue);
void onecclAllToAll(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    at::ScalarType dataType,
    xcclComm_t& comm,
    ccl::stream& xcclStream,
    void* SyclQueue);

} // namespace xccl
} // namespace c10d
