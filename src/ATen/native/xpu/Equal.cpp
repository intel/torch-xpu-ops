#include <ATen/NamedTensorUtils.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/equal_native.h>
#endif

namespace at {

bool XPUNativeFunctions::equal(const Tensor& self, const Tensor& src) {
  if (!at::namedinference::are_names_equal(
          self.unsafeGetTensorImpl(), src.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(
      self.device() == src.device(),
      "Cannot compare two tensors on "
      "different devices. Got: ",
      self.device(),
      " and ",
      src.device());
  if (self.sizes() != src.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }

  // This is the same optimization done in the cpu_equal.
  if (self.is_alias_of(src) && self.storage_offset() == src.storage_offset() &&
      self.dtype() == src.dtype() &&
      self.is_contiguous() == src.is_contiguous() &&
      self.strides().equals(src.strides()) && self.layout() == src.layout() &&
      self.is_neg() == src.is_neg() && self.is_conj() == src.is_conj()) {
    return true;
  }

  return at::XPUNativeFunctions::eq(self, src).all().item().to<bool>();
}

} // namespace at
