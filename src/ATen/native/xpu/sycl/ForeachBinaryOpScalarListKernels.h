/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <comm/xpu_aten.h>

namespace at::native::xpu {

#define FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_scalarlist__kernel

#define FOREACH_BINARY_SCALARLIST_KERNEL_NAME(NAME) \
  foreach_binary_##NAME##_scalarlist_kernel

#define FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(NAME)      \
  void FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL_NAME(NAME)( \
      TensorList tensors, at::ArrayRef<Scalar> scalars)

#define FOREACH_BINARY_SCALARLIST_KERNEL(NAME)                     \
  std::vector<Tensor> FOREACH_BINARY_SCALARLIST_KERNEL_NAME(NAME)( \
      TensorList tensors, at::ArrayRef<Scalar> scalars)

TORCH_XPU_API FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(add);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_KERNEL(add);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(sub);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_KERNEL(sub);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(mul);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_KERNEL(mul);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(div);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_KERNEL(div);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(clamp_max);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_KERNEL(clamp_max);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(clamp_min);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_KERNEL(clamp_min);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_INPLACE_KERNEL(pow);
TORCH_XPU_API FOREACH_BINARY_SCALARLIST_KERNEL(pow);

} // namespace at::native::xpu
