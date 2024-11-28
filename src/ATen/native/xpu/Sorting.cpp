
#include <ATen/MemoryOverlap.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/Sorting.h>
#include <ATen/native/TensorIterator.h>

#include <ATen/native/SortingUtils.h>
#include <ATen/native/xpu/sycl/Sorting.h>
#include <comm/TensorInfo.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/full.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/where.h>

namespace at {
namespace native {
REGISTER_XPU_DISPATCH(sort_stub, xpu::sort_stable_kernel);

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

std::tuple<Tensor&, Tensor&> median_out_xpu(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/false);
}

Tensor median_xpu(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/false);
}

std::tuple<Tensor&, Tensor&> nanmedian_out_xpu(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  return median_with_indices_impl(
      values, indices, self, dim, keepdim, /*ignore_nan=*/true);
}

Tensor nanmedian_xpu(const Tensor& self) {
  return median_impl(self, /*ignore_nan=*/true);
}

std::tuple<Tensor&, Tensor&> kthvalue_out_impl(
    Tensor& values,
    Tensor& indices,
    const Tensor& self,
    int64_t k,
    int64_t dim_,
    bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  int64_t slicesize = self.dim() == 0 ? 1 : self.size(dim);
  zero_numel_check_dims(self, dim, "kthvalue()");

  TORCH_CHECK(
      k >= 1 && k <= slicesize,
      "kthvalue(): selected number k out of range for dimension ",
      dim);

  at::assert_no_overlap(self, values);

  _reduction_with_indices_allocate_or_resize_output(
      values, indices, self, dim, keepdim);
  if (self.dim() == 0 && self.numel() == 1) {
    values.copy_(self);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }

  TORCH_CHECK(
      self.dim() <= XPU_MAX_TENSORINFO_DIMS,
      "cannot operate on more than ",
      XPU_MAX_TENSORINFO_DIMS,
      " dimensions");

  // Based on required index size, run the algorithm with the
  // appropriate index type
  if (self.numel() != 0) {
    at::native::xpu::launch_kthvalue_kernel(values, indices, self, dim, k);
  }

  if (!keepdim) {
    values.squeeze_(dim);
    indices.squeeze_(dim);
  }
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> kthvalue_out_xpu(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool keepdim,
    Tensor& values,
    Tensor& indices) {
  // See note [Writing Nondeterministic Operations]
  // If there are duplicate elements of the kth value, the procedure for
  // choosing which of the duplicates to use for the indices output is
  // nondeterministic.
  at::globalContext().alertNotDeterministic("kthvalue XPU");
  auto result = [&]() {
    NoNamesGuard guard;
    // `kthvalue_out_impl` expects contiguous in input `self`.
    return kthvalue_out_impl(
        values, indices, self.contiguous(), k, dim, keepdim);
  }();
  namedinference::propagate_names_for_reduction(values, self, dim, keepdim);
  namedinference::propagate_names_for_reduction(indices, self, dim, keepdim);
  return result;
}

} // namespace native
} // namespace at