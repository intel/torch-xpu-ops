#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/TensorNames.h>
#include <ATen/XPUNativeFunctions.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/ops/as_strided_ops.h>
#include <aten/sycl/Shape.h>
#include <comm/RegisterUtils.h>

namespace at {

Tensor XPUNativeFunctions::view(const Tensor& self, IntArrayRef size) {
  return at::native::view(self, size);
}

Tensor XPUNativeFunctions::view_as_real(const at::Tensor& self) {
  return at::native::view_as_real(self);
}

Tensor XPUNativeFunctions::view_as_complex(const Tensor& self) {
  return at::native::view_as_complex(self);
}

Tensor XPUNativeFunctions::as_strided(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    c10::optional<int64_t> storage_offset = c10::nullopt) {
  if (self.is_quantized()) {
    return at::native::as_strided_qtensorimpl(
        self, size, stride, storage_offset);
  }
  return at::native::as_strided_tensorimpl(self, size, stride, storage_offset);
}

Tensor XPUNativeFunctions::_reshape_alias(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride) {
  return at::native::_reshape_alias(self, size, stride);
}

Tensor XPUNativeFunctions::unfold(
    const Tensor& self,
    int64_t dimension,
    int64_t size,
    int64_t step) {
  return at::native::unfold(self, dimension, size, step);
}

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

inline void cat_check_no_zero_dim(const MaterializedITensorListRef& tensors) {
  size_t i = 0;
  for (const Tensor& t : tensors) {
    TORCH_CHECK(
        t.dim() > 0,
        "zero-dimensional tensor (at position ",
        i,
        ") cannot be concatenated");
    i++;
  }
}

void cat_meta(
    const ITensorListRef& tensors,
    int64_t& dim,
    Tensor& result,
    size_t& valid,
    bool& all_contiguous,
    bool& all_same_dtype,
    bool& all_same_sizes_and_stride,
    c10::MemoryFormat& memory_format) {
  // previously, size [0] tensors were the only possible empty tensors; thus, it
  // wasn't possible to cat empty tensors unless all the other tensors were
  // 1-dimensional, so we allowed these tensors to be "skipped".  We maintain
  // this behavior for backwards compatibility, but only for this specific size
  // (i.e. other empty sizes are not skipped).
  auto materialized = tensors.materialize();

  cat_check_no_zero_dim(materialized);
  dim = at::legacy_cat_wrap_dim(dim, materialized);

  // Checking names before the actual dimensions.
  auto maybe_outnames = namedinference::compute_cat_outnames(materialized);

  TORCH_CHECK(
      !materialized.empty(),
      "torch.cat(): expected a non-empty list of Tensors");

  // Look for the first valid tensor.
  valid = materialized.size();
  for (const auto i : c10::irange(materialized.size())) {
    if (!at::native::cat_should_skip_tensor(materialized[i].get())) {
      valid = i;
      break;
    }
  }

  all_contiguous = true;
  all_same_dtype = true;
  all_same_sizes_and_stride = true;
  memory_format = cat_compute_output_memory_format(materialized);

  // Compute what the output dtype should be:
  auto is_out_defined = result.defined();
  auto out_dtype = at::native::result_type(tensors);

  // If the output tensor is defined, we need to take it into account
  // when computing the actual output dtype and the flags.
  if (is_out_defined) {
    // Check for type promotion, if the output tensor is defined.
    TORCH_CHECK(
        canCast(out_dtype, result.scalar_type()),
        "torch.cat(): input types can't be cast to the desired output type ",
        result.scalar_type());
    out_dtype = result.scalar_type();
    all_contiguous = result.is_contiguous(memory_format);
  }

  // Fallback 'set_output' parameters.
  // (in case we don't find a valid tensor)
  DimVector sizes{0};
  TensorOptions options =
      materialized[0].get().options().dtype(out_dtype).memory_format(
          memory_format);

  // If we found a valid tensor, check whether the input tensors
  // are compatible, i.e. we can execute `cat` on them.
  bool found_valid_tensor = valid < materialized.size();
  if (found_valid_tensor) {
    TORCH_CHECK(
        dim <= materialized[valid].get().dim(),
        "torch.cat(): dimension ",
        dim,
        "out of range");

    // Compute the output tensor size.
    // It should have the same shape as any other valid tensor,
    // except in the dimension 'dim'.
    size_t size_at_dim = 0;
    for (const auto i : c10::irange(materialized.size())) {
      const Tensor& t = materialized[i];
      all_same_dtype = all_same_dtype && out_dtype == t.scalar_type();
      if (!at::native::cat_should_skip_tensor(t)) {
        at::native::check_cat_shape_except_dim(materialized[valid], t, dim, i);
        size_at_dim += t.size(dim);
        all_contiguous = all_contiguous && t.is_contiguous(memory_format);
        all_same_sizes_and_stride = all_same_sizes_and_stride &&
            t.sizes() == materialized[valid].get().sizes() &&
            t.strides() == materialized[valid].get().strides();
      } else {
        all_contiguous = false;
      }
    }

    // Actually set the output.
    sizes = materialized[valid].get().sizes().vec();
    sizes[dim] = size_at_dim;
    options =
        materialized[valid].get().options().dtype(out_dtype).memory_format(
            memory_format);
  }

  at::xpu::resize_out(result, sizes, {}, options);
  if (!maybe_outnames.empty()) {
    namedinference::propagate_names(result, maybe_outnames);
  }
  // Checks for overlaps between the inputs and the output tensor.
  if (is_out_defined && found_valid_tensor) {
    at::assert_no_internal_overlap(result);
    for (const Tensor& t : materialized) {
      at::assert_no_overlap(result, t);
    }
  }
}

Tensor& XPUNativeFunctions::cat_out(
    const ITensorListRef& tensors,
    int64_t dim,
    Tensor& result) {
  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, result, "xpu::cat_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, tensors, "xpu::cat_out", "tensors");

  size_t valid;
  bool all_contiguous;
  bool all_same_dtype;
  bool all_same_sizes_and_stride;
  c10::MemoryFormat memory_format;
  cat_meta(
      tensors,
      dim,
      result,
      valid,
      all_contiguous,
      all_same_dtype,
      all_same_sizes_and_stride,
      memory_format);

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
