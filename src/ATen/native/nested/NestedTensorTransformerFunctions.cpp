/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

namespace at::native {

Tensor NestedTensor_softmax_dropout_xpu(
    const Tensor& self,
    const Tensor& query) {
  std::optional<Tensor> attn_mask;

  attn_mask = NestedTensor_to_mask(query, 2, self.size(2));
  attn_mask = attn_mask->to(query.device(), /*non-blocking=*/true);
  return _masked_softmax(
      self,
      *attn_mask,
      self.dim() - 1,
      /*mask type */ 1); // NestedTensor_to_mask produces a BxT mask
}

} // namespace at::native