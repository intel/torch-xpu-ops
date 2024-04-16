#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <aten/sycl/ReduceMaxValuesKernel.h>
#include <aten/sycl/ReduceMinValuesKernel.h>
#include <aten/sycl/TensorCompare.h>
#include <comm/ReduceOpsUtils.h>

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

TensorIterator clamp_meta(
    const Tensor& self,
    const OptionalScalarRef min,
    const OptionalScalarRef max,
    Tensor& result) {
  TensorIterator iter;
  if (!min && !max) {
    TORCH_CHECK(
        false, "torch.clamp: At least one of 'min' or 'max' must not be None");
  }
  // Manual type promotion, since scalars have to participate in it
  ScalarType result_type = self.scalar_type();
  TORCH_CHECK(
      !isComplexType(result_type), "clamp is not supported for complex types");
  // Floating is the highest supported
  if (!isFloatingType(result_type)) {
    at::native::ResultTypeState state = {};
    state = at::native::update_result_type_state(self, state);

    if (min) {
      state = at::native::update_result_type_state(min.get(), state);
    }
    if (max) {
      state = at::native::update_result_type_state(max.get(), state);
    }
    result_type = at::native::result_type(state);
    // disallow type promoting inplace op
    TORCH_CHECK(
        (result_type == self.scalar_type()) ||
            (!(result.defined()) || !(result.is_same(self))),
        "result type ",
        result_type,
        " can't be cast to the desired output type ",
        self.dtype());
  }
  // make sure scalars weren't complex
  TORCH_CHECK(
      !isComplexType(result_type), "clamp is not supported for complex types");
  iter.build_unary_op(result, self.to(result_type));
  return iter;
}

Tensor& XPUNativeFunctions::clamp_out(
    const Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max,
    Tensor& result) {
  auto iter = clamp_meta(
      self,
      (min.has_value() ? at::OptionalScalarRef(&(min.value()))
                       : at::OptionalScalarRef()),
      (max.has_value() ? at::OptionalScalarRef(&(max.value()))
                       : at::OptionalScalarRef()),
      result);
  using at::native::detail::ClampLimits;
  if (min && max) {
    if ((*min).toDouble() != (*min).toDouble() ||
        (*max).toDouble() != (*max).toDouble()) {
      at::fill_(
          const_cast<Tensor&>(result),
          std::numeric_limits<double>::quiet_NaN());
    } else {
      native::xpu::clamp_scalar_kernel(iter, *min, *max);
    }
  } else if (max) {
    native::xpu::clamp_max_scalar_kernel(iter, *max);
  } else if (min) {
    native::xpu::clamp_min_scalar_kernel(iter, *min);
  }
  return result;
}

TensorIterator clamp_min_meta(
    const Tensor& self,
    const Scalar& min,
    Tensor& result) {
  TensorIterator iter;
  ScalarType result_type = self.scalar_type();
  TORCH_CHECK(
      !isComplexType(result_type), "clamp is not supported for complex types");
  TORCH_CHECK(!min.isComplex(), "clamp is not supported for complex types");
  // Floating is the highest supported
  if (!isFloatingType(result_type)) {
    auto result_type = at::native::result_type(self, min);
    TORCH_CHECK(
        (result_type == self.scalar_type() || !(result.defined()) ||
         !(result.is_same(self))),
        "result type ",
        result_type,
        " can't be cast to the desired output type ",
        self.dtype());
    iter.build_unary_op(result, self.to(result_type));
  } else {
    iter.build_borrowing_unary_op(result, self);
  }
  return iter;
}

Tensor& XPUNativeFunctions::clamp_min_out(
    const Tensor& self,
    const Scalar& min,
    Tensor& result) {
  auto iter = clamp_min_meta(self, min, result);
  if (min.toDouble() != min.toDouble()) {
    at::fill_(const_cast<Tensor&>(result), min);
  } else {
    native::xpu::clamp_min_scalar_kernel(iter, min);
  }
  return result;
}

TensorIterator clamp_max_meta(
    const Tensor& self,
    const Scalar& max,
    Tensor& result) {
  TensorIterator iter;
  // we could wrap max into tensor and send to tensor overload,
  // but relu is implemented via clamp_min, so for perf an uniformity reasons
  // do a faster but correct thing
  ScalarType result_type = self.scalar_type();
  TORCH_CHECK(
      !isComplexType(result_type), "clamp is not supported for complex types");
  TORCH_CHECK(!max.isComplex(), "clamp is not supported for complex types");
  // Floating is the highest supported
  if (!isFloatingType(result_type)) {
    auto result_type = at::native::result_type(self, max);
    TORCH_CHECK(
        (result_type == self.scalar_type()) ||
            (!(result.defined()) || !(result.is_same(self))),
        "result type ",
        result_type,
        " can't be cast to the desired output type ",
        self.dtype());
    iter.build_unary_op(result, self.to(result_type));
  } else {
    iter.build_borrowing_unary_op(result, self);
  }
  return iter;
}

Tensor& XPUNativeFunctions::clamp_max_out(
    const Tensor& self,
    const Scalar& max,
    Tensor& result) {
  auto iter = clamp_max_meta(self, max, result);
  if (max.toDouble() != max.toDouble()) {
    // TODO this is not great, building TI again is expensive, but I can't use
    // fill_stub because fill is not structured
    // this is a corner case anyway
    at::fill_(const_cast<Tensor&>(result), native::wrapped_scalar_tensor(max));
  } else {
    native::xpu::clamp_max_scalar_kernel(iter, max);
  }
  return result;
}

void min_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto iter = meta::make_reduction(
      self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  native::xpu::min_launch_kernel(iter);
}

void max_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto iter = meta::make_reduction(
      self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  native::xpu::max_launch_kernel(iter);
}

template <class Stub>
void minmax_out_impl(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    const Tensor& values,
    const Tensor& indices,
    Stub& stub) {
  NoNamesGuard guard;
  if (self.numel() > 0) {
    if (self.numel() == 1 && self.dim() == 0) {
      values.fill_(self);
      indices.fill_(0);
    } else {
      stub(values, indices, self, dim, keepdim);
    }
  }
}

static void check_unsupported_complex(const char* name, const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), name, ": does not support complex input");
}

::std::tuple<Tensor&, Tensor&> XPUNativeFunctions::min_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  dim = maybe_wrap_dim(dim, self.dim());
  at::native::zero_numel_check_dims(self, dim, "min()");
  check_unsupported_complex("min()", self);
  at::xpu::resize_reduction_with_indices(
      values, indices, self, dim, keepdim, self.scalar_type());

  minmax_out_impl(self, dim, keepdim, values, indices, min_kernel_impl);
  return {values, indices};
}

::std::tuple<Tensor&, Tensor&> XPUNativeFunctions::max_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  dim = maybe_wrap_dim(dim, self.dim());
  at::native::zero_numel_check_dims(self, dim, "max()");
  check_unsupported_complex("max()", self);
  at::xpu::resize_reduction_with_indices(
      values, indices, self, dim, keepdim, self.scalar_type());

  minmax_out_impl(self, dim, keepdim, values, indices, max_kernel_impl);
  return {values, indices};
}

} // namespace at
