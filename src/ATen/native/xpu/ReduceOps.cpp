#include <ATen/ScalarOps.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ReduceOps.h>
#include <ATen/native/xpu/sycl/ScanKernels.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>
#include <comm/ReduceOpsUtils.h>
#include <torch/library.h>

namespace at {

using namespace at::xpu;

namespace native {
REGISTER_XPU_DISPATCH(sum_stub, xpu::sum_kernel);
REGISTER_XPU_DISPATCH(mean_stub, xpu::mean_kernel);
REGISTER_XPU_DISPATCH(argmax_stub, xpu::argmax_kernel);
// REGISTER_XPU_DISPATCH(argmin_stub, xpu::argmin_kernel);
REGISTER_XPU_DISPATCH(cumsum_stub, xpu::cumsum_kernel);
} // namespace native

template <class Stub>
void impl_func_cum_ops(
    const Tensor& self,
    int64_t dim,
    const Tensor& result,
    Stub& stub) {
  NoNamesGuard guard;
  if (self.dim() == 0) {
    result.fill_(self);
  } else if (self.numel() == 0) {
    result.zero_();
  } else {
    dim = maybe_wrap_dim(dim, self.dim());
    stub(result, self.to(result.scalar_type()), dim);
  }
}

static ScalarType infer_dtype_from_optional(
    const Tensor& self,
    const optional<ScalarType>& opt_dtype,
    const Tensor& result) {
  // 'opt_dtype' has the priority for both cases.
  if (result.defined()) {
    // Otherwise, get the result type, if defined.
    return opt_dtype.value_or(result.scalar_type());
  } else {
    // Last case is to get the self type.
    // If the self type is an integer, we promote it to kLong.
    return at::native::get_dtype_from_self(self, opt_dtype, true);
  }
}

inline bool should_use_acc_buffer(at::TensorIterator& iter) {
  const auto ndim = iter.ndim();
  if (!iter.device().is_cpu() || iter.noutputs() != 1) {
    return false;
  }
  if (!at::isReducedFloatingType(iter.common_dtype())) {
    return false;
  }
  if (ndim < 2) {
    return false;
  }
  auto out_strides = iter.strides(0);
  for (const auto dim : c10::irange(0, 2)) {
    if (out_strides[dim] != 0) {
      return false;
    }
  }
  return true;
}

inline TensorIterator get_allany_iter(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef dims,
    bool keepdim) {
  return meta::make_reduction(self, result, dims, keepdim, self.scalar_type());
}

template <int identity, typename Stub>
inline void allany_impl(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef dims,
    bool keepdim,
    Stub& stub) {
  if (self.numel() == 0) {
    result.fill_(identity);
  } else if (self.numel() == 1) {
    result.copy_(self.view_as(result).to(at::kBool));
  } else {
    auto iter = get_allany_iter(self, result, dims, keepdim);
    stub(iter);
  }
}

static ScalarType get_result_or_bytebool_dtype(
    const Tensor& self,
    const Tensor& result) {
  // Refer [all, any : uint8 compatibility]
  if (result.defined()) {
    return result.scalar_type();
  } else {
    return (self.scalar_type() == kByte) ? kByte : kBool;
  }
}

static void check_result_is_bytebool(
    const char* name,
    const Tensor& self,
    const Tensor& result) {
  if (result.defined()) {
    // Refer [all, any : uint8 compatibility]
    TORCH_CHECK(
        result.scalar_type() == ScalarType::Bool ||
            result.scalar_type() == ScalarType::Byte,
        name,
        " only supports bool tensor for result, got: ",
        result.scalar_type());
  }
}

Tensor& allany_meta(
    Tensor& result,
    const char* name,
    const Tensor& self,
    OptionalIntArrayRef dims,
    bool keepdim) {
  check_result_is_bytebool(name, self, result);
  auto out_dtype = get_result_or_bytebool_dtype(self, result);
  result = resize_reduction(
      result, self, dims, keepdim, out_dtype, /*allow_empty_dims=*/true);
  return result;
}

template <class Stub>
void argmax_argmin_impl(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    const Tensor& result,
    Stub& stub) {
  c10::MaybeOwned<Tensor> in;
  DimVector dims;
  int64_t _dim = 0;

  if (dim.has_value()) {
    _dim = maybe_wrap_dim(dim.value(), self.dim());
    auto sizes = self.sizes();

    if (sizes[_dim] == 1) {
      result.fill_(0);
      return;
    }

    dims = IntArrayRef(_dim);
    in = c10::MaybeOwned<Tensor>::borrowed(self);
  } else {
    in = c10::MaybeOwned<Tensor>::owned(self.reshape({-1}));
    keepdim = false;
  }

  auto iter =
      meta::make_reduction(*in, result, dims, keepdim, self.scalar_type());

  if (iter.numel() != 0) {
    stub(iter);
  }
}

static void check_argmax_argmin(
    const char* name,
    const Tensor& self,
    const c10::optional<int64_t>& dim) {
  if (dim.has_value()) {
    auto dim_ = maybe_wrap_dim(dim.value(), self.dim());
    native::zero_numel_check_dims(self, dim_, name);
  } else {
    TORCH_CHECK_INDEX(
        self.numel() != 0,
        name,
        ": Expected reduction dim to be specified for input.numel() == 0.");
  }
}

static IntArrayRef optional_to_arrayref(const c10::optional<int64_t>& opt) {
  return opt.has_value() ? opt.value() : IntArrayRef{};
}

} // namespace at
