#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <aten/sycl/CompareKernels.h>

namespace at {

Tensor XPUNativeFunctions::eq(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::eq_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::eq_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::eq_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::eq_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::eq_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::eq(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::eq(self, wrapper);
}

Tensor& XPUNativeFunctions::eq_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::eq_(self, wrapper);
}

Tensor& XPUNativeFunctions::eq_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::eq_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::ne(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::ne_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::ne_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::ne_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::ne_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::ne_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::ne(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ne(self, wrapper);
}

Tensor& XPUNativeFunctions::ne_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ne_(self, wrapper);
}

Tensor& XPUNativeFunctions::ne_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ne_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::lt(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::lt_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::lt_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::lt_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::lt_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::lt_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::lt(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::lt(self, wrapper);
}

Tensor& XPUNativeFunctions::lt_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::lt_(self, wrapper);
}

Tensor& XPUNativeFunctions::lt_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::lt_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::le(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::le_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::le_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::le_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::le_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::le_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::le(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::le(self, wrapper);
}

Tensor& XPUNativeFunctions::le_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::le_(self, wrapper);
}

Tensor& XPUNativeFunctions::le_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::le_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::gt(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::gt_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::gt_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::gt_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::gt_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::gt_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::gt(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::gt(self, wrapper);
}

Tensor& XPUNativeFunctions::gt_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::gt_(self, wrapper);
}

Tensor& XPUNativeFunctions::gt_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::gt_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::ge(const Tensor& self, const Tensor& other) {
  Tensor out;
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::ge_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::ge_(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::comparison_op(self, self, other);
  native::xpu::ge_kernel(iter);
  return self;
}

Tensor& XPUNativeFunctions::ge_out(
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  auto iter = TensorIterator::comparison_op(out, self, other);
  native::xpu::ge_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::ge(const Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ge(self, wrapper);
}

Tensor& XPUNativeFunctions::ge_(Tensor& self, const Scalar& other) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ge_(self, wrapper);
}

Tensor& XPUNativeFunctions::ge_out(
    const Tensor& self,
    const Scalar& other,
    Tensor& out) {
  auto wrapper = native::wrapped_scalar_tensor(other);
  return XPUNativeFunctions::ge_out(self, wrapper, out);
}

Tensor XPUNativeFunctions::isnan(const Tensor& self) {
  return XPUNativeFunctions::ne(self, self);
}

Tensor& XPUNativeFunctions::isnan_out(const Tensor& self, Tensor& out) {
  return XPUNativeFunctions::ne_out(self, self, out);
}

Tensor& XPUNativeFunctions::clamp_out(
    const Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max,
    Tensor& out) {
  if (min && max) {
    auto iter = TensorIterator::unary_op(out, self);
    native::xpu::clamp_kernel(iter, *min, *max);
  } else if (max) {
    XPUNativeFunctions::clamp_max_out(self, *max, out);
  } else if (min) {
    XPUNativeFunctions::clamp_min_out(self, *min, out);
  }
  return out;
}

Tensor& XPUNativeFunctions::clamp_min_out(
    const Tensor& self,
    const Scalar& min,
    Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::clamp_min_kernel(iter, min);
  return out;
}

Tensor& XPUNativeFunctions::clamp_max_out(
    const Tensor& self,
    const Scalar& max,
    Tensor& out) {
  auto iter = TensorIterator::unary_op(out, self);
  native::xpu::clamp_max_kernel(iter, max);
  return out;
}

template <typename... Args>
Device out_device(Args&... inps) {
  for (const auto& i : {inps...}) {
    if (i.device() != at::kCPU) {
      return i.device();
    }
  }
  return at::kCPU;
}

Tensor& XPUNativeFunctions::where_out(
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

Tensor XPUNativeFunctions::where(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  auto device = out_device(condition, self, other);
  auto result_type = at::native::result_type(self, other);
  Tensor ret = at::empty({0}, self.options().dtype(result_type).device(device));
  XPUNativeFunctions::where_out(condition, self, other, ret);
  return ret;
}

} // namespace at
