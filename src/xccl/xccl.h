#pragma once

#include <ATen/xpu/XPUEvent.h>
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <oneapi/ccl.h>
#include <oneapi/ccl.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <algorithm>
#include <cstdlib>
#include <map>
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
      LOG(INFO) << "USE_CCL_V2";
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

namespace c10d {

struct XCCLStream {
  at::xpu::XPUStream xpuStream;
  ccl::stream cclStream;
  sycl::queue syclQueue;
};

using xcclComm_t = std::variant<ccl::communicator, onecclComm_t>;
using XcclDataType = std::variant<ccl::datatype, onecclDataType_t>;
using XcclRedOp = std::variant<ccl::reduction, onecclRedOp_t>;

namespace {
inline std::string getXcclVersion() {
  static std::string versionString = []() {
    bool useCCLV2 = isCCLV2EnabledCached();
    std::string versionString;
    if (useCCLV2) {
      int version = 0;
      onecclGetVersion(&version);
      const int majorBase = 10000;
      const int minorBase = 100;
      auto xcclMajor = version / majorBase;
      auto xcclMinor = (version % majorBase) / minorBase;
      auto xcclPatch =
          version % (xcclMajor * majorBase + xcclMinor * minorBase);
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
  return versionString;
}

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

const std::map<at::ScalarType, onecclDataType_t> xcclDatatypesV2 = {
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
    // use for non-reducetion op like allgather
    {at::kFloat8_e5m2, ccl::datatype::uint8},
    {at::kFloat8_e4m3fn, ccl::datatype::uint8},
    {at::kFloat8_e4m3fnuz, ccl::datatype::uint8},
    {at::kFloat8_e5m2fnuz, ccl::datatype::uint8},
};

XcclDataType getXcclDataType(
    at::ScalarType type,
    bool is_reduction_op = false) {
  if (is_reduction_op) {
    TORCH_CHECK(
        !isFloat8Type(type),
        "Float8 dtypes are not currently supported for XCCL reductions");
  }
  bool useCCLV2 = isCCLV2EnabledCached();
  if (useCCLV2) {
    auto it = xcclDatatypesV2.find(type);
    TORCH_CHECK_WITH(
        TypeError,
        it != xcclDatatypesV2.end(),
        "Input tensor data type is not supported for XCCL process group: ",
        type);
    return it->second;
  } else {
    auto it = xcclDatatypesV1.find(type);
    TORCH_CHECK_WITH(
        TypeError,
        it != xcclDatatypesV1.end(),
        "Input tensor data type is not supported for XCCL process group: ",
        type);
    return it->second;
  }
}

XcclRedOp getXcclReduceOp(const ReduceOp& reduceOp, at::Tensor& input) {
  bool useCCLV2 = isCCLV2EnabledCached();
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        if (useCCLV2)
          return onecclRedOp_t::onecclMax;
        else
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
      if (useCCLV2)
        return onecclRedOp_t::onecclSum;
      else
        return ccl::reduction::sum;
    }
#endif
    if (useCCLV2) {
      return xcclOpsV2.at(reduceOp);
    } else {
      return xcclOpsV1.at(reduceOp);
    }
  } catch (const std::out_of_range&) {
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduceOpToString(reduceOp) + " with XCCL");
  }
}
} // namespace

namespace xccl {

void oneccl_v2_group_start();
void oneccl_v2_group_end();
void onecclAllReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);
void onecclReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    const int root,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);
void onecclBroadcast(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);
void onecclReduceScatter(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const ReduceOp& reduceOp,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);
void onecclAllGather(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);
void onecclSend(
    at::Tensor& input,
    xcclComm_t& comm,
    const int dstRank,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);
void onecclRecv(
    at::Tensor& output,
    xcclComm_t& comm,
    const int srcRank,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);
void onecclGather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    xcclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);
void onecclScatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    xcclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);
void onecclAllToAll(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType _type,
    xcclComm_t& comm,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue);

} // namespace xccl
} // namespace c10d
