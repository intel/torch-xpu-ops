#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <aten/sycl/TensorCompare.h>

namespace at {

template <typename... Args>
Device out_device(Args&... inps) {
  for (const auto& i : {inps...}) {
    if (i.device() != at::kCPU) {
      return i.device();
    }
  }
  return at::kCPU;
}

Tensor& where_self_out(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  const auto result_type = at::native::result_type(self, other);
  TORCH_CHECK(
      out.scalar_type() == result_type,
      "Expected out type to be ",
      result_type,
      " but got ",
      out.scalar_type());

  auto self_ = self.scalar_type() != result_type ? self.to(result_type) : self;
  auto other_ =
      other.scalar_type() != result_type ? other.to(result_type) : other;
  auto condition_ = condition;
  auto device = out_device(condition, self_, other_);
  if (device != at::kCPU) { // allow CPU scalars on non-cpu device
    if (condition.device() != device && condition.ndimension() == 0) {
      condition_ = condition.to(device);
    }
    if (self_.device() != device && self_.ndimension() == 0) {
      self_ = self_.to(device);
    }
    if (other_.device() != device && other_.ndimension() == 0) {
      other_ = other_.to(device);
    }
  }
  if (condition_.scalar_type() == ScalarType::Byte) {
    TORCH_WARN_ONCE(
        "where received a uint8 condition tensor. This behavior is deprecated and will be removed in a future version of PyTorch. Use a boolean condition instead.");
    condition_ = condition_.to(kBool);
  }
  TORCH_CHECK(
      condition_.scalar_type() == kBool,
      "where expected condition to be a boolean tensor, but got a tensor with dtype ",
      condition_.scalar_type());
  // if there's still a device mismatch, let tensoriterator error out with it
  auto iter = at::TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(out)
                  .add_const_input(condition_)
                  .add_const_input(self_)
                  .add_const_input(other_)
                  .build();
  native::xpu::where_kernel(iter);
  return out;
}

Tensor& XPUNativeFunctions::where_out(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  return where_self_out(condition, self, other, out);
}

Tensor XPUNativeFunctions::where(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  auto device = out_device(condition, self, other);
  auto result_type = at::native::result_type(self, other);
  Tensor ret = at::empty({0}, self.options().dtype(result_type).device(device));
  where_self_out(condition, self, other, ret);
  return ret;
}

Tensor& XPUNativeFunctions::clamp_out(
    const Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max,
    Tensor& result) {
  using at::native::detail::ClampLimits;
  if (min && max) {
    if ((*min).toDouble() != (*min).toDouble() ||
        (*max).toDouble() != (*max).toDouble()) {
      at::fill_(
          const_cast<Tensor&>(result),
          std::numeric_limits<double>::quiet_NaN());
    } else {
      auto iter = TensorIterator::unary_op(result, self);
      native::xpu::clamp_scalar_kernel(iter, *min, *max);
    }
  } else if (max) {
    auto iter = TensorIterator::unary_op(result, self);
    native::xpu::clamp_max_scalar_kernel(iter, *max);
  } else if (min) {
    auto iter = TensorIterator::unary_op(result, self);
    native::xpu::clamp_min_scalar_kernel(iter, *min);
  }
  return result;
}

Tensor& XPUNativeFunctions::clamp_min_out(
    const Tensor& self,
    const Scalar& min,
    Tensor& result) {
  if (min.toDouble() != min.toDouble()) {
    at::fill_(const_cast<Tensor&>(result), min);
  } else {
    auto iter = TensorIterator::unary_op(result, self);
    native::xpu::clamp_min_scalar_kernel(iter, min);
  }
  return result;
}

Tensor& XPUNativeFunctions::clamp_max_out(
    const Tensor& self,
    const Scalar& max,
    Tensor& result) {
  if (max.toDouble() != max.toDouble()) {
    // TODO this is not great, building TI again is expensive, but I can't use
    // fill_stub because fill is not structured
    // this is a corner case anyway
    at::fill_(const_cast<Tensor&>(result), native::wrapped_scalar_tensor(max));
  } else {
    auto iter = TensorIterator::unary_op(result, self);
    native::xpu::clamp_max_scalar_kernel(iter, max);
  }
  return result;
}

} // namespace at
