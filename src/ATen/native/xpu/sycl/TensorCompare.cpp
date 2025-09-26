#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

namespace at {
namespace native {
namespace xpu {

template <typename scalar_t>
struct WhereFunctor {
  scalar_t operator()(bool cond_val, scalar_t self_val, scalar_t other_val)
      const {
    return cond_val ? self_val : other_val;
  }
};

template <typename scalar_t>
struct ClampFunctor {
  scalar_t operator()(scalar_t v, scalar_t lower, scalar_t upper) const {
    if (at::_isnan(v)) {
      return v;
    }
    if (at::_isnan(lower)) {
      return lower;
    }
    if (at::_isnan(upper)) {
      return upper;
    } else {
      return std::min(std::max(v, lower), upper);
    }
  }
};

} // namespace xpu
} // namespace native
} // namespace at
