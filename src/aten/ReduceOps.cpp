#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/ReduceMaxValuesKernel.h>
#include <aten/sycl/ReduceMinValuesKernel.h>
#include <aten/sycl/ReduceMomentKernel.h>
#include <aten/sycl/ReduceSumProdKernel.h>
#include <torch/library.h>

#include <iostream>

namespace at {

Tensor& XPUNativeFunctions::max_out(const Tensor& self, Tensor& out) {
  auto iter = at::native::make_reduction(
      "max_all", out, self, IntArrayRef{}, false, self.scalar_type());
  at::native::xpu::max_all_launch_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::max(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  XPUNativeFunctions::max_out(self, result);
  return result;
}

::std::tuple<Tensor&, Tensor&> XPUNativeFunctions::max_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& max,
    Tensor& max_values) {
  auto iter = meta::make_reduction(
      self, max, max_values, dim, keepdim, self.scalar_type(), kLong);
  at::native::xpu::max_launch_kernel(iter);
  return {max, max_values};
}

Tensor& XPUNativeFunctions::min_out(const Tensor& self, Tensor& out) {
  auto iter = at::native::make_reduction(
      "min_all", out, self, IntArrayRef{}, false, self.scalar_type());
  at::native::xpu::min_all_launch_kernel(iter);
  return out;
}

Tensor XPUNativeFunctions::min(const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  XPUNativeFunctions::min_out(self, result);
  return result;
}

::std::tuple<Tensor&, Tensor&> XPUNativeFunctions::min_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& min,
    Tensor& min_indices) {
  auto iter = meta::make_reduction(
      self, min, min_indices, dim, keepdim, self.scalar_type(), kLong);
  at::native::xpu::min_launch_kernel(iter);
  return {min, min_indices};
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
      at::native::xpu::sum_kernel(iter);
    }
  }
  return result;
}

Tensor XPUNativeFunctions::sum(
    const Tensor& self,
    OptionalIntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = at::native::get_dtype_from_self(self, opt_dtype, true);
  Tensor result = at::native::create_reduction_result(
      self, dim.value_or(IntArrayRef{}), keepdim, dtype);
  return XPUNativeFunctions::sum_out(self, dim, keepdim, dtype, result);
}

Tensor& XPUNativeFunctions::sum_out(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    Tensor& out) {
  return XPUNativeFunctions::sum_out(self, IntArrayRef{}, false, dtype, out);
}

Tensor& XPUNativeFunctions::mean_out(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype,
    Tensor& result) {
  ScalarType dtype = result.scalar_type();
  // TODO: the TensorIterator reduction implementation of mean
  // (mean_kernel_impl()) is unvectorized and leads to very poor performance
  // for production workloads. Once that's fixed, the following code can be used
  // in lieu of the sum + divide implementation below.
  if (self.device().is_cpu()) {
    int64_t dim_prod = 1;
    if (!opt_dim.has_value() || opt_dim.value().empty() ||
        self.ndimension() == 0) {
      dim_prod = self.numel();
    } else {
      auto dim = opt_dim.value();
      for (auto d : dim) {
        dim_prod *= self.size(d);
      }
    }
    auto& result_mut = const_cast<Tensor&>(result);
    // For accuracy reasons, BF16/FP16 mean should be computed via the
    // following approach:
    //  cast_fp32 -> sum -> div -> cast_bf16_or_fp16
    //
    // Such an approach is necessary because if we were to choose the same
    // approach for BF16/FP16 as FP32 here, then it would have resulted in
    // the following code-flow -
    // cast_fp32 -> sum -> cast_bf16 -> cast_fp32 -> div -> cast_bf16,
    // which, in turn, does not produce as accurate results.
    bool is_half_type = (dtype == kHalf || dtype == kBFloat16);
    auto sum_out_dtype = is_half_type ? ScalarType::Float : dtype;
    result_mut = is_half_type ? result_mut.to(sum_out_dtype) : result_mut;
    // If dtype is FP16 or BF16, self (input tensor) will initially be cast to
    // FP32 in sum_out. This results in having to read that FP32 tensor again,
    // but maybe in the future, we could revise the implementation to not
    // materialize that intermediate FP32 tensor. That approach would probably
    // require some modifications in binary_kernel_reduce_vec(),
    // TensorIteratorBase::for_each(), and
    // TensorIteratorBase::serial_for_each(), apart from sum kernel for CPU.
    at::sum_out(result_mut, self, opt_dim, keepdim, sum_out_dtype)
        .div_(dim_prod);
    // After sum & div, cast result_mut back to BF16 or FP16, if required.
    result_mut = is_half_type ? result_mut.to(dtype) : result_mut;
  } else {
    // device is not CPU
    auto iter = at::meta::make_reduction_from_out_ty(
        self, result, opt_dim, keepdim, dtype);
    if (iter.numel() == 0) {
      result.fill_(std::numeric_limits<double>::quiet_NaN());
    } else {
      at::native::xpu::mean_kernel(iter);
    }
  }
  return result;
}

Tensor XPUNativeFunctions::mean(
    const Tensor& self,
    OptionalIntArrayRef opt_dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype) {
  ScalarType dtype = at::native::get_dtype_from_self(self, opt_dtype, true);
  Tensor result = at::native::create_reduction_result(
      self, opt_dim.value_or(IntArrayRef{}), keepdim, dtype);
  return XPUNativeFunctions::mean_out(self, opt_dim, keepdim, dtype, result);
}

} // namespace at
