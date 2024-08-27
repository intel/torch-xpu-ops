#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Sorting.h>
#include <ATen/xpu/XPUNativeFunctions.h>
#include <comm/RegisterUtils.h>
#include <comm/TensorInfo.h>

namespace at {

void sort_stable_meta(
    const Tensor& self,
    Tensor& values,
    Tensor& indices,
    int64_t dim) {
  maybe_wrap_dim(dim, self.dim());

  // See issue: https://github.com/pytorch/pytorch/issues/65863
  // Strides should be dense, so as not to allocate too much memory.
  // We either use 'self' strides, or infer dense strides from them.
  std::vector<int64_t> strides = (self.is_non_overlapping_and_dense())
      ? self.strides().vec()
      : at::infer_dense_strides(self.sizes(), self.strides());
  auto sizes = self.sizes();
  if (values.defined()) {
    at::xpu::resize_out(values, sizes, strides, self.options());
  } else {
    values = at::xpu::create_out(sizes, strides, self.options());
  }
  if (indices.defined()) {
    at::xpu::resize_out(indices, sizes, strides, self.options().dtype(kLong));
  } else {
    indices = at::xpu::create_out(sizes, strides, self.options().dtype(kLong));
  }
}

::std::tuple<Tensor, Tensor> XPUNativeFunctions::sort(
    const Tensor& self,
    ::std::optional<bool> stable,
    int64_t dim,
    bool descending) {
  Tensor values, indices;
  sort_stable_meta(self, values, indices, dim);
  return native::xpu::sort_stable_kernel(
      self, stable, values, indices, dim, descending);
}

::std::tuple<Tensor&, Tensor&> XPUNativeFunctions::sort_out(
    const Tensor& self,
    ::std::optional<bool> stable,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, values, "xpu::sort_out_values_stable", "values");
  c10::impl::check_and_update_common_device(
      common_device, indices, "xpu::sort_out_values_stable", "indices");
  c10::impl::check_and_update_common_device(
      common_device, self, "xpu::sort_out_values_stable", "self");
  sort_stable_meta(self, values, indices, dim);
  return native::xpu::sort_stable_kernel(
      self, stable, values, indices, dim, descending);
}

Tensor XPUNativeFunctions::argsort(
    const Tensor& self,
    bool stable,
    int64_t dim,
    bool descending) {
  Tensor values, indices;
  sort_stable_meta(self, values, indices, dim);
  return std::get<1>(native::xpu::sort_stable_kernel(
      self, stable, values, indices, dim, descending));
}

std::tuple<Tensor&, Tensor&> median_with_indices_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    bool ignore_nan) {
  // See note [Writing Nondeterministic Operations]
  // If there are duplicate elements of a median value, the procedure for
  // choosing which of the duplicates to use for the indices output is
  // nondeterministic.
  at::globalContext().alertNotDeterministic("median XPU with indices output");
  NoNamesGuard guard;

  dim = at::maybe_wrap_dim(dim, self.dim());
  Tensor in = self.dim() > 0 ? self.contiguous() : self.unsqueeze(0);

  checkDeviceType("median", {values, indices}, self.device().type());
  checkScalarType("median", {indices, "indices", 1}, kLong);
  checkSameType("median", {values, "values", 0}, {self, "self", 2});

  TORCH_CHECK(
      self.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "median() cannot operate on more than ",
      XPU_MAX_TENSORINFO_DIMS,
      " dimensions");

  std::vector<int64_t> out_shape = self.sizes().vec();
  native::zero_numel_check_dims(self, dim, "median()");
  if (self.dim() > 0) {
    assert(dim >= 0);
    assert(dim < static_cast<int64_t>(out_shape.size()));

    if (keepdim) {
      out_shape[dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + dim);
    }
  }

  values.resize_(out_shape);
  indices.resize_(out_shape);

  // Only launch kernel for non-empty tensors
  if (self.numel() > 0) {
    // Ensure #dim is the same for all tensors required for reduction
    Tensor vals = keepdim && self.dim() > 0 ? values : values.unsqueeze(dim);
    Tensor inds = keepdim && self.dim() > 0 ? indices : indices.unsqueeze(dim);

    at::native::xpu::launch_median_kernel(vals, inds, in, dim, ignore_nan);
  }

  guard.reset();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);

  return std::forward_as_tuple(values, indices);
}

Tensor median_impl(const Tensor& self, bool ignore_nan) {
  NoNamesGuard guard;

  int64_t size = self.numel();
  // Return nan for empty tensors
  if (size <= 0) {
    return at::full({}, std::numeric_limits<float>::quiet_NaN())
        .to(self.options());
  }

  // Sort input tensor to efficiently query for median element
  Tensor sorted = std::get<0>(self.flatten().sort());

  if (!ignore_nan) {
    // For torch.median return either the middle element or nan (sorted as
    // largest) if there are any
    int64_t k = (size - 1) / 2;
    return at::where(sorted[-1].isnan(), sorted[-1], sorted[k]);
  } else {
    // For torch.nanmedian return the middle element among the non-nan values
    int64_t k = ((size - 1) - sorted.isnan().sum().item<int64_t>()) / 2;
    return sorted[k].clone(); // Clone so we aren't keeping `sorted` alive
  }
}

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::median_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/false);
}

Tensor XPUNativeFunctions::median(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/false);
}

std::tuple<Tensor&, Tensor&> XPUNativeFunctions::nanmedian_out(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/true);
}

Tensor XPUNativeFunctions::nanmedian(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/true);
}

} // namespace at
