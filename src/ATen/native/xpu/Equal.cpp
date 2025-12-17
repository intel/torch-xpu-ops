/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/NamedTensorUtils.h>

#include <ATen/ops/equal_native.h>

namespace at {
namespace xpu {
// Note:
// Seems {op}_xpu_dispatch.h is not generated in codegen via
// backendwhitelist mode. We have to manually add a declaration here.
at::Tensor eq(const at::Tensor& self, const at::Tensor& other);
} // namespace xpu
namespace native {
bool xpu_equal(const Tensor& self, const Tensor& src) {
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

  return at::xpu::eq(self, src).all().item().to<bool>();
}
} // namespace native
} // namespace at
