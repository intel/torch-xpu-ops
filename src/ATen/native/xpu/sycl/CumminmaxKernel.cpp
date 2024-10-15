#include <ATen/Dispatch.h>

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/xpu/sycl/ScanKernels.h>
#include <ATen/native/xpu/sycl/ScanUtils.h>

namespace at::native::xpu {

static c10::MaybeOwned<Tensor> contiguous_out_arg(const Tensor& tensor) {
  if (tensor.is_contiguous()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
  return c10::MaybeOwned<Tensor>::owned(
      at::empty(tensor.sizes(), tensor.options()));
}

void launch_cummax_kernel(
    const Tensor& self,
    const Tensor& values,
    const Tensor& indices,
    int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool,
      ScalarType::Half,
      ScalarType::BFloat16,
      self.scalar_type(),
      "cummax_xpu",
      [&]() {
        scalar_t init = self.is_floating_point()
            ? -std::numeric_limits<scalar_t>::infinity()
            : std::numeric_limits<scalar_t>::lowest();
        scan_with_indices<INCLUSIVE_TYPE, scalar_t, scalar_t, int64_t>(
            self, values, indices, dim, init, std::greater_equal<scalar_t>());
      });
}

void launch_cummin_kernel(
    const Tensor& self,
    const Tensor& values,
    const Tensor& indices,
    int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND3(
      ScalarType::Bool,
      ScalarType::Half,
      ScalarType::BFloat16,
      self.scalar_type(),
      "cummin_xpu",
      [&]() {
        scalar_t init = self.is_floating_point()
            ? std::numeric_limits<scalar_t>::infinity()
            : std::numeric_limits<scalar_t>::max();
        scan_with_indices<INCLUSIVE_TYPE, scalar_t, scalar_t, int64_t>(
            self, values, indices, dim, init, std::less_equal<scalar_t>());
      });
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

} // namespace at::native::xpu
