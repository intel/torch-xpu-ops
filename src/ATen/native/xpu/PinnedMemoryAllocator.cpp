/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/CPUFunctions.h>

#include <ATen/xpu/PinnedMemoryAllocator.h>
#include <comm/xpu_aten.h>

#include <ATen/ops/is_pinned_native.h>

namespace at {
namespace native {
// Note: The user must call is_pinned(device='xpu') to explicitly call here.
bool is_pinned_xpu(const Tensor& self, std::optional<Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !device.has_value() || device->type() == c10::DeviceType::XPU);

  return at::detail::getXPUHooks().isPinnedPtr(self.storage().data());
}

// Note: The user must call tensor.pin_memory(device='xpu') to explicitly call
// here.
Tensor _pin_memory_xpu(const Tensor& self, std::optional<Device> device) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !device.has_value() || device->type() == c10::DeviceType::XPU);

  auto* allocator = at::xpu::getPinnedMemoryAllocator();
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(),
      at::detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options())
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace native
} // namespace at
