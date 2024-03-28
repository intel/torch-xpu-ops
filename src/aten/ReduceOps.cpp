#include <ATen/ATen.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/sycl/ReduceArgMaxKernel.h>
#include <aten/sycl/ReduceLogicKernel.h>
#include <aten/sycl/ReduceMomentKernel.h>
#include <aten/sycl/ReduceSumProdKernel.h>
#include <torch/library.h>

#include <iostream>

namespace at {

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

Tensor XPUNativeFunctions::any(const Tensor& self, int64_t dim, bool keepdim) {
  Tensor result;
  auto result_type = get_result_or_bytebool_dtype(self, result);
  result = at::empty({0}, self.options().dtype(result_type));
  allany_impl<0>(self, result, dim, keepdim, native::xpu::or_kernel);
  return result;
}

Tensor& XPUNativeFunctions::any_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& out) {
  allany_impl<0>(self, out, dim, keepdim, native::xpu::or_kernel);
  return out;
}

Tensor& XPUNativeFunctions::any_out(const Tensor& self, Tensor& out) {
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

Tensor& XPUNativeFunctions::argmax_out(
    const Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    Tensor& out) {
  argmax_argmin_impl(self, dim, keepdim, out, native::xpu::argmax_kernel);
  return out;
}

} // namespace at
