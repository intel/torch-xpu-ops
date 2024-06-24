#include <ATen/ATen.h>
#include <ATen/ScalarOps.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#include <ATen/native/xpu/sycl/ReduceMaxValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceMinValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceOpsKernels.h>
#include <ATen/native/xpu/sycl/ScanKernels.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>
#include <comm/ReduceOpsUtils.h>

namespace at {

using namespace at::xpu;

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

Tensor& XPUNativeFunctions::cumsum_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype,
    Tensor& result) {
  // Checking whether 'dim' is valid.
  maybe_wrap_dim(dim, self.dim());

  ScalarType out_dtype;

  if (!result.defined()) {
    auto is_integral =
        at::isIntegralType(self.scalar_type(), /*includeBool=*/true);
    out_dtype =
        dtype.value_or(is_integral ? ScalarType::Long : self.scalar_type());
    result = at::empty_strided(
        self.sizes(), self.strides(), self.options().dtype(out_dtype));
  } else {
    at::native::resize_output(result, self.sizes());
    result.as_strided_(self.sizes(), self.strides());
  }

  impl_func_cum_ops(self, dim, result, at::native::xpu::cumsum_kernel);
  return result;
}

Tensor XPUNativeFunctions::cumsum(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  Tensor result;
  return cumsum_out(self, dim, dtype, result);
}

Tensor& XPUNativeFunctions::cumsum_(
    Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  return cumsum_out(self, dim, dtype, self);
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

Tensor& XPUNativeFunctions::sum_out(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype,
    Tensor& result) {
  auto out_dtype = infer_dtype_from_optional(self, opt_dtype, result);
  result = resize_reduction(result, self, opt_dim, keepdim, out_dtype);
  auto iter = meta::make_reduction_from_out_ty(
      self, result, opt_dim, keepdim, result.scalar_type());
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    // Here is a limitation of TensorIterator reductions for permuted input with
    // lower precision on CPU. Consider the case: TensorIterator coalesces such
    // input and output to >= 2 dims tensors, and the output stride is [0, 0, x,
    // x, ...] with x >= 0 (two reduced dimensions and non-reduced dims). Since
    // the reduction loop only operates on two dimensions at a time, the
    // intermediate sums is forced to do accumulation in the second reduced dim
    // with lower precision. See https://github.com/pytorch/pytorch/issues/83149
    if (should_use_acc_buffer(iter)) {
      auto tmp_output =
          at::empty(result.sizes(), result.options().dtype(kFloat));
      at::sum_outf(
          self.to(ScalarType::Float),
          opt_dim,
          keepdim,
          /*dtype=*/c10::nullopt,
          tmp_output);
      result.copy_(tmp_output);
    } else {
      native::xpu::sum_kernel(iter);
    }
  }
  return result;
}

Tensor XPUNativeFunctions::sum(
    const Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype) {
  Tensor out;
  return XPUNativeFunctions::sum_out(self, dim, keepdim, opt_dtype, out);
}

Tensor& mean_meta(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    optional<ScalarType> opt_dtype,
    Tensor& out) {
  auto in_dtype = at::native::get_dtype_from_self(self, opt_dtype, true);
  if (!at::isFloatingType(in_dtype) && !at::isComplexType(in_dtype)) {
    std::string what = "Input";
    std::string dtype = toString(self.scalar_type());

    if (opt_dtype.has_value()) {
      what = "Optional";
      dtype = toString(opt_dtype.value());
    }

    TORCH_CHECK(
        false,
        "mean(): could not infer output dtype. ",
        what,
        " dtype must be either a floating point or complex dtype. ",
        "Got: ",
        dtype);
  }

  auto out_dtype = infer_dtype_from_optional(self, opt_dtype, out);
  out = resize_reduction(out, self, opt_dim, keepdim, out_dtype);
  return out;
}

Tensor& XPUNativeFunctions::mean_out(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype,
    Tensor& result) {
  result = mean_meta(self, opt_dim, keepdim, opt_dtype, result);
  ScalarType dtype = result.scalar_type();
  // device is not CPU
  auto iter = at::meta::make_reduction_from_out_ty(
      self, result, opt_dim, keepdim, dtype);
  if (iter.numel() == 0) {
    result.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    native::xpu::mean_kernel(iter);
  }
  return result;
}

Tensor XPUNativeFunctions::mean(
    const Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim,
    ::std::optional<at::ScalarType> dtype) {
  Tensor out;
  out = mean_meta(self, dim, keepdim, dtype, out);
  out = XPUNativeFunctions::mean_out(self, dim, keepdim, dtype, out);
  return out;
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

// aten::all.dim
Tensor XPUNativeFunctions::all(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor out;
  out = allany_meta(out, "all", self, dim, keepdim);
  allany_impl<1>(self, out, dim, keepdim, native::xpu::and_kernel);
  return out;
}

// aten::all.out
Tensor& XPUNativeFunctions::all_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& out) {
  out = allany_meta(out, "all", self, dim, keepdim);
  allany_impl<1>(self, out, dim, keepdim, native::xpu::and_kernel);
  return out;
}

// aten::all.dims
Tensor XPUNativeFunctions::all(
    const Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim) {
  Tensor out;
  out = allany_meta(out, "all", self, dim, keepdim);
  allany_impl<1>(self, out, dim, keepdim, native::xpu::and_kernel);
  return out;
}

// aten::all.dims_out
Tensor& XPUNativeFunctions::all_out(
    const Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim,
    Tensor& out) {
  out = allany_meta(out, "all", self, dim, keepdim);
  allany_impl<1>(self, out, dim, keepdim, native::xpu::and_kernel);
  return out;
}

// aten::all
Tensor XPUNativeFunctions::all(const Tensor& self) {
  Tensor out;
  out = allany_meta(out, "all", self, {}, false);
  allany_impl<1>(self, out, {}, false, native::xpu::and_kernel);
  return out;
}

// aten::all.all_out
Tensor& XPUNativeFunctions::all_out(const Tensor& self, Tensor& out) {
  out = allany_meta(out, "all", self, {}, false);
  allany_impl<1>(self, out, {}, false, native::xpu::and_kernel);
  return out;
}

// aten::any.dim
Tensor XPUNativeFunctions::any(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor out;
  out = allany_meta(out, "any", self, dim, keepdim);
  allany_impl<0>(self, out, dim, keepdim, native::xpu::or_kernel);
  return out;
}

// aten::any.out
Tensor& XPUNativeFunctions::any_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& out) {
  out = allany_meta(out, "any", self, dim, keepdim);
  allany_impl<0>(self, out, dim, keepdim, native::xpu::or_kernel);
  return out;
}

// aten::any.dims
Tensor XPUNativeFunctions::any(
    const Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim) {
  Tensor out;
  out = allany_meta(out, "any", self, dim, keepdim);
  allany_impl<0>(self, out, dim, keepdim, native::xpu::or_kernel);
  return out;
}

// aten::any.dims_out
Tensor& XPUNativeFunctions::any_out(
    const Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim,
    Tensor& out) {
  out = allany_meta(out, "any", self, dim, keepdim);
  allany_impl<0>(self, out, dim, keepdim, native::xpu::or_kernel);
  return out;
}

// aten::any
Tensor XPUNativeFunctions::any(const Tensor& self) {
  Tensor out;
  out = allany_meta(out, "any", self, {}, false);
  allany_impl<0>(self, out, {}, false, native::xpu::or_kernel);
  return out;
}

// aten::any.any_out
Tensor& XPUNativeFunctions::any_out(const Tensor& self, Tensor& out) {
  out = allany_meta(out, "any", self, {}, false);
  allany_impl<0>(self, out, {}, false, native::xpu::or_kernel);
  return out;
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

Tensor& argmax_meta(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    Tensor& out) {
  check_argmax_argmin("argmax()", self, dim);
  return resize_reduction(out, self, optional_to_arrayref(dim), keepdim, kLong);
}

Tensor& XPUNativeFunctions::argmax_out(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    Tensor& out) {
  out = argmax_meta(self, dim, keepdim, out);
  argmax_argmin_impl(self, dim, keepdim, out, native::xpu::argmax_kernel);
  return out;
}

Tensor XPUNativeFunctions::argmax(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim) {
  Tensor out;
  out = argmax_meta(self, dim, keepdim, out);
  argmax_argmin_impl(self, dim, keepdim, out, native::xpu::argmax_kernel);
  return out;
}

static Tensor amax_amin_meta(
    Tensor& result,
    const char* name,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  if (result.defined()) {
    TORCH_CHECK(
        self.scalar_type() == result.scalar_type(),
        "Expected the dtype for input and out to match, but got ",
        self.scalar_type(),
        " for input's dtype and ",
        result.scalar_type(),
        " for out's dtype.");
  }
  if (self.numel() == 0) {
    at::native::zero_numel_check_dims(self, dim, "amax()");
  }
  const ScalarType& out_dtype =
      result.defined() ? result.scalar_type() : self.scalar_type();
  return resize_reduction(result, self, dim, keepdim, out_dtype);
}

template <class Stub>
void amax_amin_impl(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    const Tensor& result,
    Stub& stub) {
  auto iter =
      meta::make_reduction(self, result, dim, keepdim, self.scalar_type());

  if (iter.numel() != 0) {
    stub(iter);
  }
}

Tensor& XPUNativeFunctions::amax_out(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    Tensor& out) {
  out = amax_amin_meta(out, "amax()", self, dim, keepdim);
  amax_amin_impl(self, dim, keepdim, out, native::xpu::max_all_kernel);
  return out;
}

Tensor XPUNativeFunctions::amax(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  Tensor out;
  out = amax_amin_meta(out, "amax()", self, dim, keepdim);
  amax_amin_impl(self, dim, keepdim, out, native::xpu::max_all_kernel);
  return out;
}

Tensor& XPUNativeFunctions::amin_out(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    Tensor& out) {
  out = amax_amin_meta(out, "amin()", self, dim, keepdim);
  amax_amin_impl(self, dim, keepdim, out, native::xpu::min_all_kernel);
  return out;
}

Tensor XPUNativeFunctions::amin(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim) {
  Tensor out;
  out = amax_amin_meta(out, "amin()", self, dim, keepdim);
  amax_amin_impl(self, dim, keepdim, out, native::xpu::min_all_kernel);
  return out;
}

} // namespace at
