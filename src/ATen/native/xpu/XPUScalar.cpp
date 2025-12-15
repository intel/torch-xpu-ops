/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch_v2.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/_local_scalar_dense_native.h>
#include <comm/SYCLContext.h>

namespace at::native {

Scalar _local_scalar_dense_xpu(const Tensor& self) {
  Scalar r;
  AT_DISPATCH_V2(
      self.scalar_type(),
      "_local_scalar_dense",
      AT_WRAP([&] {
        auto value = at::detail::empty_cpu(
            {1}, /* size */
            c10::CppTypeToScalarType<scalar_t>(), /* dtype */
            std::nullopt, /* layout */
            std::nullopt, /* device */
            false, /* pin_memory */
            std::nullopt /* memory format */
        );

        auto queue = at::xpu::getCurrentSYCLQueue();
        auto e = queue.memcpy(
            (void*)value.const_data_ptr<scalar_t>(),
            self.const_data_ptr<scalar_t>(),
            sizeof(scalar_t));
        e.wait();

        r = Scalar(*value.const_data_ptr<scalar_t>());
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      AT_EXPAND(AT_FLOAT8_TYPES),
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
  return r;
}

} // namespace at::native
