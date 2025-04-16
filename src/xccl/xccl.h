#include <oneapi/ccl.h>
#include <oneapi/ccl.hpp>

#if defined(CCL_MAJOR_VERSION) &&  \
    ((CCL_MAJOR_VERSION > 2021) || \
     (CCL_MAJOR_VERSION == 2021) && (CCL_MINOR_VERSION >= 15))
#define XCCL_HAS_AVG 1
#endif // oneCCL version >= 2021.15

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

namespace c10d {

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

const std::map<at::ScalarType, ccl::datatype> xcclDatatypesV2 = {
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

auto getXcclDataType(at::ScalarType type, bool is_reduction_op = false) {
  if (is_reduction_op) {
    TORCH_CHECK(
        !isFloat8Type(type),
        "Float8 dtypes are not currently supported for XCCL reductions");
  }
  bool useCCLV2 = isCCLV2EnabledCached();
  auto it = useCCLV2 ? xcclDatatypesV2.find(type) : xcclDatatypesV1.find(type);
  TORCH_CHECK_WITH(
      TypeError,
      (it != xcclDatatypesV2.end() && it != xcclDatatypesV1.end()),
      "Input tensor data type is not supported for XCCL process group: ",
      type);
  return it->second;
}

auto getXcclReduceOp(const ReduceOp& reduceOp, at::Tensor& input) {
  bool useCCLV2 = isCCLV2EnabledCached();
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        return useCCLV2 ? onecclRedOp_t::onecclMax : ccl::reduction::max;
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
      return useCCLV2 ? onecclRedOp_t::onecclSum : ccl::reduction::sum;
    }
#endif
    return useCCLV2 ? xcclOpsV2.at(reduceOp) : xcclOpsV1.at(reduceOp);
  } catch (const std::out_of_range&) {
    C10_THROW_ERROR(
        ValueError,
        "Cannot use ReduceOp." + reduceOpToString(reduceOp) + " with XCCL");
  }
}
} // namespace c10d

namespace xccl {

//   xcclSum = 0
// xcclProd = 1
// xcclMax = 3
// xcclMin = 2
// xcclAvg = 4
// xcclNumOps = 5

// xcclInt8 = 0
// xcclChar = 0
// xcclUint8 = 1
// xcclInt32 = 4
// xcclInt = 4
// xcclUint32 = 5
// xcclInt64 = 6
// xcclUint64 = 7
// xcclFloat16 = 8
// xcclHalf = 8
// xcclFloat32 = 9
// xcclFloat = 9
// xcclFloat64 = 10
// xcclDouble = 10
// xcclBfloat16 = 11
// xcclNumTypes = 11
// /* Reduction operation selector */
// enum class xcclRedOp { Sum = 0, Prod = 1, Max = 2, Min = 3, Avg = 4, NumOps =
// 5 };

// /* Data types */
// enum class xcclDataType {
//   Int8 = 0,
//   Char = 0,
//   Uint8 = 1,
//   Int32 = 2,
//   Int = 2,
//   Uint32 = 3,
//   Int64 = 4,
//   Uint64 = 5,
//   Float16 = 6,
//   Half = 6,
//   Float32 = 7,
//   Float = 7,
//   Float64 = 8,
//   Double = 8,
//   Bfloat16 = 9,
//   NumTypes = 10
// };

void onecclGroupStart() {
  if (isCCLV2EnabledCached()) {
    onecclGroupStart();
  } else {
    ccl::group_start();
  }
}

void onecclGroupEnd() {
  if (isCCLV2EnabledCached()) {
    onecclGroupEnd();
  } else {
    ccl::group_end();
  }
}

void onecclReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    ReduceOp& reduceOp,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  auto xcclDataType = getXcclDataType(input.scalar_type(), true);
  auto xcclReduceOp = getXcclReduceOp(reduceOp, input);
  if (isCCLV2EnabledCached()) {
    onecclReduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        xcclReduceOp,
        comm,
        &SyclQueue);
  } else {
    ccl::allreduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        xcclReduceOp,
        comm,
        xcclStream);
  }
  return;
}

void onecclReduce(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    ReduceOp& reduceOp,
    const int root,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  auto xcclDataType = getXcclDataType(input.scalar_type(), true);
  auto xcclReduceOp = getXcclReduceOp(reduceOp, input);
  if (isCCLV2EnabledCached()) {
    onecclReduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        xcclReduceOp,
        root,
        comm,
        &SyclQueue);
  } else {
    ccl::reduce(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        xcclReduceOp,
        root,
        comm,
        xcclStream);
  }
  return;
}

void onecclBroadcast(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    const int root,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  auto xcclDataType = getXcclDataType(input.scalar_type());
  if (isCCLV2EnabledCached()) {
    onecclBroadcast(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        root,
        comm,
        &SyclQueue);
  } else {
    ccl::broadcast(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        root,
        comm,
        xcclStream);
  }
  return;
}

void onecclReduceScatter(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    ReduceOp& reduceOp,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  auto xcclDataType = getXcclDataType(input.scalar_type(), true);
  auto xcclReduceOp = getXcclReduceOp(reduceOp, input);
  if (isCCLV2EnabledCached()) {
    onecclReduceScatter(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)output.numel(),
        xcclDataType,
        xcclReduceOp,
        comm,
        &SyclQueue);
  } else {
    ccl::reduce_scatter(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)output.numel(),
        xcclDataType,
        xcclReduceOp,
        comm,
        xcclStream);
  }
  return;
}

void onecclAllGather(
    at::Tensor& input,
    at::Tensor& output,
    xcclComm_t& comm,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  auto xcclDataType = getXcclDataType(input.scalar_type());
  if (isCCLV2EnabledCached()) {
    onecclAllGather(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        comm,
        &SyclQueue);
  } else {
    ccl::allgather(
        input.data_ptr(),
        output.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        comm,
        xcclStream);
  }
  return;
}

void onecclSend(
    at::Tensor& input,
    xcclComm_t& comm,
    const int dstRank,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  auto xcclDataType = getXcclDataType(input.scalar_type());
  if (isCCLV2EnabledCached()) {
    onecclSend(
        input.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        dstRank,
        comm,
        &SyclQueue);
  } else {
    ccl::send(
        input.data_ptr(),
        (size_t)input.numel(),
        xcclDataType,
        dstRank,
        comm,
        xcclStream);
  }
  return;
}

void onecclRecv(
    at::Tensor& output,
    xcclComm_t& comm,
    const int srcRank,
    at::xpu::XPUStream& stream,
    ccl::stream& xcclStream,
    sycl::queue& SyclQueue) {
  auto xcclDataType = getXcclDataType(output.scalar_type());
  if (isCCLV2EnabledCached()) {
    onecclRecv(
        output.data_ptr(),
        (size_t)output.numel(),
        xcclDataType,
        srcRank,
        comm,
        &SyclQueue);
  } else {
    ccl::recv(
        output.data_ptr(),
        (size_t)output.numel(),
        xcclDataType,
        srcRank,
        comm,
        xcclStream);
  }
  return;
}

} // namespace xccl
