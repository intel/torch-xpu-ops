#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/MaxMinElementwiseKernels.h>
#include <ATen/native/xpu/sycl/ReduceMaxValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceMinValuesKernels.h>
#include <ATen/native/xpu/sycl/TensorCompareKernels.h>
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

Tensor& clamp_out_impl(
    const Tensor& self,
    TensorIteratorBase& iter,
    const OptionalScalarRef min,
    const OptionalScalarRef max,
    Tensor& result) {
  using at::native::detail::ClampLimits;
  if (min && max) {
    if (min.get().toDouble() != min.get().toDouble() ||
        max.get().toDouble() != max.get().toDouble()) {
      at::fill_(
          const_cast<Tensor&>(result),
          std::numeric_limits<double>::quiet_NaN());
    } else {
      native::xpu::clamp_scalar_kernel(iter, min.get(), max.get());
    }
  } else if (max) {
    native::xpu::clamp_max_scalar_kernel(iter, max.get());
  } else if (min) {
    native::xpu::clamp_min_scalar_kernel(iter, min.get());
  }
  return result;
}

Tensor XPUNativeFunctions::clamp(
    const Tensor& self,
    const ::std::optional<at::Scalar>& min,
    const ::std::optional<at::Scalar>& max) {
  auto min_ =
      (min.has_value() ? at::OptionalScalarRef(&(min.value()))
                       : at::OptionalScalarRef());
  auto max_ =
      (max.has_value() ? at::OptionalScalarRef(&(max.value()))
                       : at::OptionalScalarRef());
  Tensor result;
  auto iter = clamp_meta(self, min_, max_, result);
  result = clamp_out_impl(self, iter, min_, max_, result);
  return iter.output();
}

Tensor& XPUNativeFunctions::clamp_out(
    const Tensor& self,
    const ::std::optional<Scalar>& min,
    const ::std::optional<Scalar>& max,
    Tensor& result) {
  auto min_ =
      (min.has_value() ? at::OptionalScalarRef(&(min.value()))
                       : at::OptionalScalarRef());
  auto max_ =
      (max.has_value() ? at::OptionalScalarRef(&(max.value()))
                       : at::OptionalScalarRef());
  auto iter = clamp_meta(self, min_, max_, result);
  result = clamp_out_impl(self, iter, min_, max_, result);
  return result;
}

Tensor& XPUNativeFunctions::clamp_(
    Tensor& self,
    const ::std::optional<at::Scalar>& min,
    const ::std::optional<at::Scalar>& max) {
  auto min_ =
      (min.has_value() ? at::OptionalScalarRef(&(min.value()))
                       : at::OptionalScalarRef());
  auto max_ =
      (max.has_value() ? at::OptionalScalarRef(&(max.value()))
                       : at::OptionalScalarRef());
  auto iter = clamp_meta(self, min_, max_, self);
  self = clamp_out_impl(self, iter, min_, max_, self);
  return self;
}

TensorIterator clamp_tensor_meta(
    const Tensor& self,
    const OptionalTensorRef min,
    const OptionalTensorRef max,
    Tensor& result) {
  TensorIterator iter;
  TORCH_CHECK(
      min || max,
      "torch.clamp: At least one of 'min' or 'max' must not be None");
  TORCH_CHECK(
      !isComplexType(self.scalar_type()),
      "clamp is not supported for complex types");
#define CLAMP_CONFIG()                      \
  TensorIteratorConfig()                    \
      .set_check_mem_overlap(true)          \
      .add_output(result)                   \
      .add_const_input(self)                \
      .promote_inputs_to_common_dtype(true) \
      .cast_common_dtype_to_outputs(true)   \
      .enforce_safe_casting_to_output(true)

  if (min && max) {
    iter.build(CLAMP_CONFIG().add_const_input(*min).add_const_input(*max));
  } else if (min) {
    iter.build(CLAMP_CONFIG().add_const_input(*min));
  } else if (max) {
    iter.build(CLAMP_CONFIG().add_const_input(*max));
  }
  return iter;
}

Tensor& clamp_tensor_out_impl(
    const Tensor& self,
    TensorIteratorBase& iter,
    const OptionalTensorRef min,
    const OptionalTensorRef max,
    Tensor& result) {
  if (min && max) {
    native::xpu::clamp_kernel(iter);
  } else if (min) {
    native::xpu::maximum_kernel(iter);
  } else if (max) {
    native::xpu::minimum_kernel(iter);
  }
  return result;
}

Tensor XPUNativeFunctions::clamp(
    const Tensor& self,
    const ::std::optional<at::Tensor>& min,
    const ::std::optional<at::Tensor>& max) {
  auto min_ =
      ((min.has_value() && (*min).defined()) ? at::OptionalTensorRef(*min)
                                             : at::OptionalTensorRef());
  auto max_ =
      ((max.has_value() && (*max).defined()) ? at::OptionalTensorRef(*max)
                                             : at::OptionalTensorRef());
  Tensor result;
  auto iter = clamp_tensor_meta(self, min_, max_, result);
  result = clamp_tensor_out_impl(self, iter, min_, max_, result);
  return iter.output();
}

Tensor& XPUNativeFunctions::clamp_out(
    const Tensor& self,
    const ::std::optional<at::Tensor>& min,
    const ::std::optional<at::Tensor>& max,
    Tensor& result) {
  auto min_ =
      ((min.has_value() && (*min).defined()) ? at::OptionalTensorRef(*min)
                                             : at::OptionalTensorRef());
  auto max_ =
      ((max.has_value() && (*max).defined()) ? at::OptionalTensorRef(*max)
                                             : at::OptionalTensorRef());
  auto iter = clamp_tensor_meta(self, min_, max_, result);
  result = clamp_tensor_out_impl(self, iter, min_, max_, result);
  return result;
}

Tensor& XPUNativeFunctions::clamp_(
    Tensor& self,
    const ::std::optional<at::Tensor>& min,
    const ::std::optional<at::Tensor>& max) {
  auto min_ =
      ((min.has_value() && (*min).defined()) ? at::OptionalTensorRef(*min)
                                             : at::OptionalTensorRef());
  auto max_ =
      ((max.has_value() && (*max).defined()) ? at::OptionalTensorRef(*max)
                                             : at::OptionalTensorRef());
  auto iter = clamp_tensor_meta(self, min_, max_, self);
  self = clamp_tensor_out_impl(self, iter, min_, max_, self);
  return self;
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

Tensor& clamp_max_out_impl(
    const Tensor& self,
    TensorIteratorBase& iter,
    const Scalar& max,
    Tensor& result) {
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

Tensor XPUNativeFunctions::clamp_max(const Tensor& self, const Scalar& max) {
  Tensor result;
  auto iter = clamp_max_meta(self, max, result);
  result = clamp_max_out_impl(self, iter, max, result);
  return iter.output();
}

Tensor& XPUNativeFunctions::clamp_max_out(
    const Tensor& self,
    const Scalar& max,
    Tensor& result) {
  auto iter = clamp_max_meta(self, max, result);
  result = clamp_max_out_impl(self, iter, max, result);
  return result;
}

Tensor& XPUNativeFunctions::clamp_max_(Tensor& self, const Scalar& max) {
  auto iter = clamp_max_meta(self, max, self);
  self = clamp_max_out_impl(self, iter, max, self);
  return self;
}

TensorIterator clamp_max_tensor_meta(
    const Tensor& self,
    const Tensor& max,
    Tensor& result) {
  TensorIterator iter;
  iter.build_borrowing_binary_op(result, self, max);
  return iter;
}

Tensor XPUNativeFunctions::clamp_max(const Tensor& self, const Tensor& max) {
  Tensor result;
  auto iter = clamp_max_tensor_meta(self, max, result);
  native::xpu::minimum_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::clamp_max_out(
    const Tensor& self,
    const Tensor& max,
    Tensor& result) {
  auto iter = clamp_max_tensor_meta(self, max, result);
  native::xpu::minimum_kernel(iter);
  return result;
}

Tensor& XPUNativeFunctions::clamp_max_(Tensor& self, const Tensor& max) {
  auto iter = clamp_max_tensor_meta(self, max, self);
  native::xpu::minimum_kernel(iter);
  return self;
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

Tensor& clamp_min_out_impl(
    const Tensor& self,
    TensorIteratorBase& iter,
    const Scalar& min,
    Tensor& result) {
  if (min.toDouble() != min.toDouble()) {
    at::fill_(const_cast<Tensor&>(result), min);
  } else {
    native::xpu::clamp_min_scalar_kernel(iter, min);
  }
  return result;
}

Tensor XPUNativeFunctions::clamp_min(const Tensor& self, const Scalar& min) {
  Tensor result;
  auto iter = clamp_min_meta(self, min, result);
  result = clamp_min_out_impl(self, iter, min, result);
  return iter.output();
}

Tensor& XPUNativeFunctions::clamp_min_out(
    const Tensor& self,
    const Scalar& min,
    Tensor& result) {
  auto iter = clamp_min_meta(self, min, result);
  result = clamp_min_out_impl(self, iter, min, result);
  return result;
}

Tensor& XPUNativeFunctions::clamp_min_(Tensor& self, const Scalar& min) {
  auto iter = clamp_min_meta(self, min, self);
  self = clamp_min_out_impl(self, iter, min, self);
  return self;
}

TensorIterator clamp_min_tensor_meta(
    const Tensor& self,
    const Tensor& min,
    Tensor& result) {
  TensorIterator iter;
  iter.build_borrowing_binary_op(result, self, min);
  return iter;
}

Tensor XPUNativeFunctions::clamp_min(const Tensor& self, const Tensor& min) {
  Tensor result;
  auto iter = clamp_min_tensor_meta(self, min, result);
  native::xpu::maximum_kernel(iter);
  return iter.output();
}

Tensor& XPUNativeFunctions::clamp_min_out(
    const Tensor& self,
    const Tensor& min,
    Tensor& result) {
  auto iter = clamp_min_tensor_meta(self, min, result);
  native::xpu::maximum_kernel(iter);
  return result;
}

Tensor& XPUNativeFunctions::clamp_min_(Tensor& self, const Tensor& min) {
  auto iter = clamp_min_tensor_meta(self, min, self);
  native::xpu::maximum_kernel(iter);
  return self;
}

void min_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto iter = meta::make_reduction(
      self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  native::xpu::min_kernel(iter);
}

void max_kernel_impl(
    const Tensor& result,
    const Tensor& indice,
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  auto iter = meta::make_reduction(
      self, result, indice, dim, keepdim, self.scalar_type(), kLong);
  native::xpu::max_kernel(iter);
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

std::tuple<Tensor, Tensor> XPUNativeFunctions::_aminmax(
    const Tensor& self,
    int64_t dim,
    bool keepdim) {
  return XPUNativeFunctions::aminmax(self, dim, keepdim);
}

} // namespace at
