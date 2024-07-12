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

static void cum_ops_meta(
    const char* name,
    const Tensor& self,
    int64_t dim,
    std::optional<ScalarType> dtype,
    Tensor& result) {
  // Checking whether 'dim' is valid.
  maybe_wrap_dim(dim, self.dim());

  ScalarType out_dtype;

  if (result.defined()) {
    out_dtype = dtype.value_or(result.scalar_type());
    at::xpu::resize_out(
        result,
        self.sizes(),
        {},
        self.options().dtype(out_dtype));
  } else {
    auto is_integral = at::isIntegralType(self.scalar_type(), /*includeBool=*/true);
    out_dtype = dtype.value_or(is_integral ? ScalarType::Long : self.scalar_type());
    result = at::xpu::create_out(
        self.sizes(),
        {},
        self.options().dtype(out_dtype));
  }

  namedinference::propagate_names(result, self);
}

Tensor& XPUNativeFunctions::cumsum_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype,
    Tensor& result) {
  cum_ops_meta("cumsum", self, dim, dtype, result);

  impl_func_cum_ops(self, dim, result, at::native::xpu::cumsum_kernel);
  return result;
}

Tensor XPUNativeFunctions::cumsum(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  Tensor result;
  return XPUNativeFunctions::cumsum_out(self, dim, dtype, result);
}

Tensor& XPUNativeFunctions::cumsum_(
    Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  return XPUNativeFunctions::cumsum_out(self, dim, dtype, self);
}

Tensor& XPUNativeFunctions::cumprod_out(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype,
    Tensor& result) {
  cum_ops_meta("cumprod", self, dim, dtype, result);

  impl_func_cum_ops(self, dim, result, at::native::xpu::cumprod_kernel);
  return result;
}

Tensor XPUNativeFunctions::cumprod(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  Tensor result;
  return XPUNativeFunctions::cumprod_out(self, dim, dtype, result);
}

Tensor& XPUNativeFunctions::cumprod_(
    Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  return XPUNativeFunctions::cumprod_out(self, dim, dtype, self);
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

static inline void warn_invalid_degrees_of_freedom(
    const char* fname,
    const TensorIterator& iter,
    double correction) {
  int64_t reducing_over_num_elements = iter.num_output_elements() == 0
      ? 0
      : iter.numel() / iter.num_output_elements();
  if (reducing_over_num_elements - correction <= 0) {
    TORCH_WARN(
        fname,
        "(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel).");
  }
}

static Tensor& std_var_out(
    const char* fname,
    Tensor& result,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction_opt,
    bool keepdim,
    bool take_sqrt) {
  TORCH_CHECK(
      self.device().is_xpu(),
      "std and var only supports tensors on an XPU device, but got: ",
      self.device().type());
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      "std and var only supports strided layout, got: ",
      self.layout());
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) ||
          at::isComplexType(self.scalar_type()),
      "std and var only support floating point and complex dtypes");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate variance of real and imaginary components
    // separately then add to get overall variance.
    ScalarType dtype =
        c10::toRealValueType(at::native::get_dtype_from_result(result, {}));
    Tensor real_in = at::real(self);
    Tensor real_out = at::empty({0}, self.options().dtype(dtype));
    std_var_out(
        fname,
        real_out,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    Tensor imag_in = at::imag(self);
    Tensor imag_out = at::empty({0}, self.options().dtype(dtype));
    std_var_out(
        fname,
        imag_out,
        imag_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    at::add_out(result, real_out, imag_out);
    if (take_sqrt) {
      at::sqrt_out(result, result);
    }
    return result;
  }

  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = at::native::get_dtype_from_result(result, {});
  auto iter =
      at::native::make_reduction(fname, result, self, dim, keepdim, dtype);
  TORCH_CHECK(
      at::canCast(self.scalar_type(), result.scalar_type()),
      "result type ",
      self.scalar_type(),
      " can't be cast to the "
      "desired output type ",
      result.scalar_type());
  warn_invalid_degrees_of_freedom(fname, iter, correction);

  if (iter.numel() == 0) {
    // Trivial reduction
    result.fill_(std::numeric_limits<double>::quiet_NaN());
    return result;
  } else {
    native::xpu::std_var_kernel(iter, correction, take_sqrt);
  }
  return result;
}

static std::tuple<Tensor&, Tensor&> std_var_mean_out(
    const char* fname,
    Tensor& result1,
    Tensor& result2,
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction_opt,
    bool keepdim,
    bool take_sqrt) {
  AT_ASSERT(result1.defined() && result2.defined());
  TORCH_CHECK(
      self.device().is_xpu(),
      fname,
      " only supports tensors on an XPU device, got: ",
      self.device().type());
  TORCH_CHECK(
      self.layout() == Layout::Strided,
      fname,
      " only supports strided layout, got: ",
      self.layout());
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()) ||
          at::isComplexType(self.scalar_type()),
      fname,
      " only support floating point and complex dtypes");
  TORCH_CHECK(
      result1.scalar_type() == c10::toRealValueType(result2.scalar_type()),
      fname,
      " expected result1 to be real and match the precision of result2. Got ",
      result1.scalar_type(),
      " and ",
      result2.scalar_type(),
      ".");

  if (at::isComplexType(self.scalar_type())) {
    // For complex, calculate for real and imaginary components separately then
    // combine as: variance = var_real + var_imag mean = mean_real + j *
    // mean_imag
    ScalarType dtype =
        c10::toRealValueType(at::native::get_dtype_from_result(result1, {}));
    Tensor real_in = at::real(self);
    Tensor real_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor real_out_mean = at::empty({0}, self.options().dtype(dtype));
    std_var_mean_out(
        fname,
        real_out_var,
        real_out_mean,
        real_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    Tensor imag_in = at::imag(self);
    Tensor imag_out_var = at::empty({0}, self.options().dtype(dtype));
    Tensor imag_out_mean = at::empty({0}, self.options().dtype(dtype));
    std_var_mean_out(
        fname,
        imag_out_var,
        imag_out_mean,
        imag_in,
        dim,
        correction_opt,
        keepdim,
        /*take_sqrt=*/false);

    at::add_out(result1, real_out_var, imag_out_var);
    if (take_sqrt) {
      at::sqrt_out(result1, result1);
    }
    at::complex_out(result2, real_out_mean, imag_out_mean);
    return std::tuple<Tensor&, Tensor&>(result1, result2);
  }

  // Computation for floating point
  const auto correction = correction_opt.value_or(1).toDouble();
  ScalarType dtype = at::native::get_dtype_from_result(result1, {});
  auto iter = at::native::make_reduction(
      fname, result1, result2, self, dim, keepdim, dtype);
  warn_invalid_degrees_of_freedom(fname, iter, correction);

  if (iter.numel() == 0) {
    // Trivial reduction
    result1.fill_(std::numeric_limits<double>::quiet_NaN());
    result2.fill_(std::numeric_limits<double>::quiet_NaN());
  } else {
    native::xpu::std_var_kernel(iter, correction, take_sqrt);
  }
  return std::tuple<Tensor&, Tensor&>(result1, result2);
}

static inline TensorOptions options_to_value_type(TensorOptions opts) {
  auto scalar_type = typeMetaToScalarType(opts.dtype());
  return opts.dtype(c10::toRealValueType(scalar_type));
}

Tensor XPUNativeFunctions::std(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& XPUNativeFunctions::std_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result) {
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& XPUNativeFunctions::var_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result) {
  return std_var_out("var", result, self, dim, correction, keepdim, false);
}

Tensor XPUNativeFunctions::var(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return std_var_out("var", result, self, dim, correction, keepdim, false);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::var_mean(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "var_mean", result1, result2, self, dim, correction, keepdim, false);
}

std::tuple<Tensor, Tensor> XPUNativeFunctions::std_mean(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "std_mean", result1, result2, self, dim, correction, keepdim, true);
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
