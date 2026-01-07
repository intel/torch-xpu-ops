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

#define FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(NAME) \
  std::vector<Tensor> foreach_##NAME##_kernel(       \
      TensorList input,                              \
      TensorList tensors1,                           \
      TensorList tensors2,                           \
      at::ArrayRef<Scalar> scalars)

#define FOREACH_POINTWISE_OP_SCALARLIST_INPLACE_KERNEL(NAME) \
  void foreach_##NAME##_kernel_(                             \
      TensorList input,                                      \
      TensorList tensors1,                                   \
      TensorList tensors2,                                   \
      at::ArrayRef<Scalar> scalars)

TORCH_XPU_API FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(addcmul);
TORCH_XPU_API FOREACH_POINTWISE_OP_SCALARLIST_INPLACE_KERNEL(addcmul);
TORCH_XPU_API FOREACH_POINTWISE_OP_SCALARLIST_KERNEL(addcdiv);
TORCH_XPU_API FOREACH_POINTWISE_OP_SCALARLIST_INPLACE_KERNEL(addcdiv);

} // namespace at::native::xpu
