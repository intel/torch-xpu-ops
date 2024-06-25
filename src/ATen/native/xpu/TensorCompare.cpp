#include <ATen/ScalarOps.h>
#include <ATen/TensorIndexing.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/xpu/sycl/ReduceMaxValuesKernel.h>
#include <ATen/native/xpu/sycl/ReduceMinValuesKernel.h>
#include <ATen/native/xpu/sycl/TensorCompare.h>
#include <comm/ReduceOpsUtils.h>

#include <ATen/ops/result_type_native.h>

namespace at {

namespace native {
// REGISTER_XPU_DISPATCH(where_fn, xpu::where_kernel);
REGISTER_XPU_DISPATCH(where_kernel, xpu::where_kernel);
REGISTER_XPU_DISPATCH(clamp_min_scalar_stub, xpu::clamp_min_scalar_kernel);
REGISTER_XPU_DISPATCH(clamp_max_scalar_stub, xpu::clamp_max_scalar_kernel);
} // namespace native

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

} // namespace at
