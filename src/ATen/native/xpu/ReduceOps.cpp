
#include <ATen/NamedTensorUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>

#include <ATen/native/DispatchStub.h>
#include <ATen/native/Fill.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/ScanKernels.h>
#include <ATen/native/xpu/sycl/ReduceMaxValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceMinValuesKernels.h>
#include <ATen/native/xpu/sycl/ReduceOpsKernels.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>
#include <comm/ReduceOpsUtils.h>
#include <torch/library.h>

#include <ATen/ops/add.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/real.h>
#include <ATen/ops/sqrt.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(sum_stub, &xpu::sum_kernel);
REGISTER_XPU_DISPATCH(mean_stub, &xpu::mean_kernel);
REGISTER_XPU_DISPATCH(prod_stub, &xpu::prod_kernel);
REGISTER_XPU_DISPATCH(argmax_stub, &xpu::argmax_kernel);
REGISTER_XPU_DISPATCH(argmin_stub, &xpu::argmin_kernel);
REGISTER_XPU_DISPATCH(and_stub, &xpu::and_kernel);
REGISTER_XPU_DISPATCH(or_stub, &xpu::or_kernel);
REGISTER_XPU_DISPATCH(max_values_stub, &xpu::max_values_kernel);
REGISTER_XPU_DISPATCH(min_values_stub, &xpu::min_values_kernel);
REGISTER_XPU_DISPATCH(std_var_stub, &xpu::std_var_kernel);
REGISTER_XPU_DISPATCH(cumsum_stub, &xpu::cumsum_kernel);
REGISTER_XPU_DISPATCH(cumprod_stub, &xpu::cumprod_kernel);
REGISTER_XPU_DISPATCH(nansum_stub, &xpu::nansum_kernel);

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

Tensor std_xpu(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& std_xpu_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result) {
  return std_var_out("std", result, self, dim, correction, keepdim, true);
}

Tensor& var_xpu_out(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim,
    Tensor& result) {
  return std_var_out("var", result, self, dim, correction, keepdim, false);
}

Tensor var_xpu(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result = at::empty({0}, options_to_value_type(self.options()));
  return std_var_out("var", result, self, dim, correction, keepdim, false);
}

std::tuple<Tensor, Tensor> var_mean_xpu(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "var_mean", result1, result2, self, dim, correction, keepdim, false);
}

std::tuple<Tensor, Tensor> std_mean_xpu(
    const Tensor& self,
    at::OptionalIntArrayRef dim,
    const std::optional<Scalar>& correction,
    bool keepdim) {
  Tensor result1 = at::empty({0}, options_to_value_type(self.options()));
  Tensor result2 = at::empty({0}, self.options());
  return std_var_mean_out(
      "std_mean", result1, result2, self, dim, correction, keepdim, true);
}

void cummax_helper_xpu(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  at::native::xpu::cummax_kernel(self, values, indices, dim);
}

void cummin_helper_xpu(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  at::native::xpu::cummin_kernel(self, values, indices, dim);
}

Tensor& _logcumsumexp_out_xpu(const Tensor& self, int64_t dim, Tensor& result) {
  return at::native::xpu::logcumsumexp_kernel(self, dim, result);
}

Tensor _logcumsumexp_xpu(const Tensor& self, int64_t dim) {
  Tensor result = at::empty_like(self, MemoryFormat::Contiguous);
  return _logcumsumexp_out_xpu(self, dim, result);
}

void aminmax_impl(
    const Tensor& self,
    int64_t dim_opt,
    bool keepdim,
    Tensor& min,
    Tensor& max) {
  auto dtype = self.scalar_type();
  TensorIterator iter =
      make_reduction("aminmax_xpu", min, max, self, dim_opt, keepdim, dtype);
  if (iter.numel() != 0) {
    native::xpu::aminmax_kernel(iter);
  }
}

void aminmax_allreduce_impl(const Tensor& self, Tensor& min, Tensor& max) {
  auto dtype = self.scalar_type();
  auto iter = make_reduction(
      "aminmax_xpu", min, max, self, IntArrayRef{}, false, dtype);
  TORCH_CHECK(
      iter.numel() > 0, "min_max on a tensor with no elements is not defined.");
  native::xpu::aminmax_allreduce_kernel(iter);
}

REGISTER_XPU_DISPATCH(aminmax_stub, &aminmax_impl);
REGISTER_XPU_DISPATCH(aminmax_allreduce_stub, &aminmax_allreduce_impl)

} // namespace native
} // namespace at
