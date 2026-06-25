/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#define CCL_ENABLE_ZE
#define CCL_ENABLE_SYCL

#include <comm/Macros.h>
DISABLE_SYCL_DEPRECATED_WARNING_BEGIN
// Official suppression macro provided by Intel SYCL headers for
// host-only compilation (without -fsycl).
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include <ATen/xpu/XPUEvent.h>
#undef SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
DISABLE_SYCL_DEPRECATED_WARNING_END
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <oneapi/ccl.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/logger.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#if defined(CCL_MAJOR_VERSION) &&  \
    ((CCL_MAJOR_VERSION > 2022) || \
     (CCL_MAJOR_VERSION == 2022) && (CCL_MINOR_VERSION >= 0))
#define ENABLE_XCCL_PREMUL_SUM_SUPPORT
#endif // oneCCL version >= 2022.0

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

inline const std::string& getVersionString() {
  static std::string versionString = []() {
    int version = 0;
    onecclGetVersion(&version);
    const int majorBase = 10000;
    const int minorBase = 100;
    auto xcclMajor = version / majorBase;
    auto xcclMinor = (version % majorBase) / minorBase;
    auto xcclPatch =
        version % (xcclMajor * majorBase + xcclMinor * minorBase);
    return std::to_string(xcclMajor) + "." + std::to_string(xcclMinor) + "." +
        std::to_string(xcclPatch);
  }();
  return versionString;
}

namespace c10d {

struct XCCLStream {
  at::xpu::XPUStream xpuStream;
};

struct xcclComm_t {
  onecclComm_t onecclComm{nullptr};

  xcclComm_t() = default;
  explicit xcclComm_t(onecclComm_t comm) : onecclComm(comm) {}
};

const std::map<c10d::ReduceOp, onecclRedOp_t> xcclOpsV2 = {
    {ReduceOp::MIN, onecclRedOp_t::onecclMin},
    {ReduceOp::MAX, onecclRedOp_t::onecclMax},
    {ReduceOp::SUM, onecclRedOp_t::onecclSum},
    {ReduceOp::PRODUCT, onecclRedOp_t::onecclProd},
    {ReduceOp::AVG, onecclRedOp_t::onecclAvg},
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

namespace {

struct XCCLTraitsV2 {
  using OpType = onecclRedOp_t;
  using CommType = onecclComm_t;

#if defined(ENABLE_XCCL_PREMUL_SUM_SUPPORT)
  static void destroyOp(OpType op, CommType comm) {
    onecclRedOpDestroy(op, comm);
  }
#endif
};

template <typename Traits>
struct xcclRedOpRAII {
  using OpType = typename Traits::OpType;
  using CommType = typename Traits::CommType;

  xcclRedOpRAII() = default;
  xcclRedOpRAII(OpType op) : op_(op) {}
  xcclRedOpRAII(OpType op, CommType comm)
      : op_(op), comm_(comm), premul_sum_(true) {}

  xcclRedOpRAII(const xcclRedOpRAII&) = delete;
  xcclRedOpRAII& operator=(const xcclRedOpRAII&) = delete;

  xcclRedOpRAII(xcclRedOpRAII&& tmp) noexcept : xcclRedOpRAII() {
    std::swap(tmp.op_, this->op_);
    std::swap(tmp.comm_, this->comm_);
    std::swap(tmp.premul_sum_, this->premul_sum_);
  }

#if defined(ENABLE_XCCL_PREMUL_SUM_SUPPORT)
  ~xcclRedOpRAII() {
    if (premul_sum_ && comm_) {
      Traits::destroyOp(op_, comm_);
    }
  }
#endif

  operator OpType() const {
    return op_;
  }

  OpType op_{};
  CommType comm_{};
  bool premul_sum_ = false;
};

using xcclRedOpRAIIV2 = xcclRedOpRAII<XCCLTraitsV2>;

#ifdef ENABLE_XCCL_PREMUL_SUM_SUPPORT
template <typename T, onecclDataType_t dataType>
inline xcclRedOpRAIIV2 unpackPreMulSumV2(
    const ReduceOp& reduceOp,
    onecclComm_t comm) {
  const auto* preMulSupplement =
      reinterpret_cast<PreMulSumSupplement*>(reduceOp.supplement_.get());
  onecclRedOp_t preMulSum{};
  bool has_tensor = preMulSupplement->tensor_factor.defined();
  auto residence = has_tensor
      ? onecclScalarResidence_t::onecclScalarDevice
      : onecclScalarResidence_t::onecclScalarHostImmediate;
  const T* ptr_factor = has_tensor
      ? preMulSupplement->tensor_factor.const_data_ptr<T>()
      : nullptr;
  T scalar_factor = T(preMulSupplement->double_factor);
  onecclRedOpCreatePreMulSum(
      &preMulSum,
      /*scalar=*/has_tensor ? const_cast<T*>(ptr_factor) : &scalar_factor,
      dataType,
      residence,
      comm);
  return xcclRedOpRAIIV2(preMulSum, comm);
}
#endif // ENABLE_XCCL_PREMUL_SUM_SUPPORT

inline onecclDataType_t getXcclDataTypeV2(
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

inline xcclRedOpRAIIV2 getXcclReduceOpV2(
    const ReduceOp& reduceOp,
    at::Tensor& input,
    const onecclDataType_t& dataType,
    onecclComm_t comm) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        return xcclRedOpRAIIV2(onecclRedOp_t::onecclMax);
      }
      if (reduceOp == ReduceOp::AVG) {
        C10_THROW_ERROR(
            TypeError, "Cannot use ReduceOp.AVG with boolean inputs");
      }
    }
    if (reduceOp == ReduceOp::PREMUL_SUM) {
#ifdef ENABLE_XCCL_PREMUL_SUM_SUPPORT
      switch (dataType) {
        case onecclDataType_t::onecclFloat16:
          return unpackPreMulSumV2<at::Half, onecclDataType_t::onecclFloat16>(
              reduceOp, comm);
        case onecclDataType_t::onecclFloat32:
          return unpackPreMulSumV2<float, onecclDataType_t::onecclFloat32>(
              reduceOp, comm);
        case onecclDataType_t::onecclBfloat16:
          return unpackPreMulSumV2<float, onecclDataType_t::onecclBfloat16>(
              reduceOp, comm);
        case onecclDataType_t::onecclFloat64:
          return unpackPreMulSumV2<double, onecclDataType_t::onecclFloat64>(
              reduceOp, comm);
        default:
          C10_THROW_ERROR(
              TypeError,
              "PreMulSum Data type must be half, float, bfloat16 or double");
      }
#else
      C10_THROW_ERROR(ValueError, "PreMulSum requires oneCCL>=2022.0");
#endif // ENABLE_XCCL_PREMUL_SUM_SUPPORT
    }
    return xcclRedOpRAIIV2(xcclOpsV2.at(reduceOp));
  } catch (const std::out_of_range&) {
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduceOpToString(reduceOp) + " with XCCL");
  }
}

inline void broadcastUniqueXCCLID(
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

} // namespace

namespace xccl {

void oneccl_group_start();
void oneccl_group_end();
void onecclAllReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    at::xpu::XPUStream& stream);
void onecclReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    const int root,
    at::xpu::XPUStream& stream);
void onecclBroadcast(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream);
void onecclReduceScatter(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const c10d::ReduceOp& reduceOp,
    at::xpu::XPUStream& stream);
void onecclAllGather(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    at::xpu::XPUStream& stream);
void onecclSend(
    at::Tensor& input,
    xcclComm_t& comm,
    const int dstRank,
    at::xpu::XPUStream& stream);
void onecclRecv(
    at::Tensor& output,
    xcclComm_t& comm,
    const int srcRank,
    at::xpu::XPUStream& stream);
void onecclGather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    xcclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream);
void onecclScatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    xcclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream);
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
    at::xpu::XPUStream& stream);

} // namespace xccl
} // namespace c10d
