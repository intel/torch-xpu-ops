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

namespace {

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
