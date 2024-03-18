#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorNames.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <aten/sycl/CatKernel.h>

namespace at {

namespace impl {

inline c10::MemoryFormat cat_compute_output_memory_format(
    const MaterializedITensorListRef& inputs) {
  c10::optional<c10::MemoryFormat> format = c10::nullopt;
  for (const Tensor& t : inputs) {
    auto f = t.suggest_memory_format();
    if (f == c10::MemoryFormat::Contiguous) {
      return f;
    }
    if (format.has_value() && format.value() != f) {
      return c10::MemoryFormat::Contiguous;
    }
    format = f;
  }
  return format.value();
}

} // namespace impl

Tensor& XPUNativeFunctions::cat_out(
    const ITensorListRef& tensors,
    int64_t dim,
    Tensor& result) {
  auto materialized = tensors.materialize();
  dim = at::legacy_cat_wrap_dim(dim, materialized);

  // Look for the first valid tensor.
  size_t valid = materialized.size();
  for (const auto i : c10::irange(materialized.size())) {
    if (!at::native::cat_should_skip_tensor(materialized[i].get())) {
      valid = i;
      break;
    }
  }

  bool all_same_dtype = true;
  bool all_same_sizes_and_stride = true;
  auto memory_format = impl::cat_compute_output_memory_format(materialized);
  bool all_contiguous = result.is_contiguous(memory_format);
  auto out_dtype = result.scalar_type();

  for (const auto i : c10::irange(materialized.size())) {
    const Tensor& t = materialized[i];
    all_same_dtype = all_same_dtype && out_dtype == t.scalar_type();
    if (!at::native::cat_should_skip_tensor(t)) {
      all_contiguous = all_contiguous && t.is_contiguous(memory_format);
      all_same_sizes_and_stride = all_same_sizes_and_stride &&
          t.sizes() == materialized[valid].get().sizes() &&
          t.strides() == materialized[valid].get().strides();
    } else {
      all_contiguous = false;
    }
  }

  at::native::xpu::cat_out_kernel(
      tensors,
      dim,
      valid,
      all_contiguous,
      all_same_dtype,
      all_same_sizes_and_stride,
      memory_format,
      result);

  return result;
}

} // namespace at
