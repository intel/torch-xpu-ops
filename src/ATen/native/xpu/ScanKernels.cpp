#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/xpu/ScanKernels.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>

namespace at::native::xpu {

static c10::MaybeOwned<Tensor> contiguous_out_arg(const Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
  return c10::MaybeOwned<Tensor>::owned(
      at::empty(tensor.sizes(), tensor.options()));
}

void cumsum_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto result_ = contiguous_out_arg(result);

  launch_cumsum_kernel(*result_, self, dim);

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

void cumprod_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto result_ = contiguous_out_arg(result);

  launch_cumprod_kernel(*result_, self, dim);

  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

void cummax_kernel(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  TensorArg output_arg{values, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{self, "input", 3};
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  auto values_ = contiguous_out_arg(values);
  auto indices_ = contiguous_out_arg(indices);
  launch_cummax_kernel(self, *values_, *indices_, dim);
  if (!values.is_same(*values_)) {
    values.copy_(*values_);
  }
  if (!indices.is_same(*indices_)) {
    indices.copy_(*indices_);
  }
}

void cummin_kernel(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  TensorArg output_arg{values, "output", 1};
  TensorArg indices_arg{indices, "indices", 2};
  TensorArg input_arg{self, "input", 3};
  checkAllSameGPU(__func__, {output_arg, indices_arg, input_arg});

  auto values_ = contiguous_out_arg(values);
  auto indices_ = contiguous_out_arg(indices);
  launch_cummin_kernel(self, *values_, *indices_, dim);
  if (!values.is_same(*values_)) {
    values.copy_(*values_);
  }
  if (!indices.is_same(*indices_)) {
    indices.copy_(*indices_);
  }
}

Tensor& logcumsumexp_kernel(const Tensor& self, int64_t dim, Tensor& result) {
  const auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  result.resize_(self.sizes());
  if (self.dim() == 0) {
    result.fill_(self);
    return result;
  }
  if (self.numel() == 0) {
    result.zero_();
    return result;
  }

  TensorArg output_arg{result, "output", 1};
  TensorArg input_arg{self, "input", 2};
  checkAllSameGPU(__func__, {output_arg, input_arg});

  auto result_ = contiguous_out_arg(result);
  launch_logcumsumexp_kernel(*result_, self, wrap_dim);
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

} // namespace at::native::xpu