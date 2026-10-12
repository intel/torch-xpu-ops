/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * SPDX-License-Identifier: Apache-2.0
 */

// Direct test of at::native::zero_xpu_ (the device-memset implementation of
// zero_). It calls the native function directly (bypassing the aten::zero_
// dispatcher), so it can be verified without relying on the upstream,
// coordinated `XPU: zero_xpu_` dispatch registration.
//
// Verifies zero_xpu_ clears dtypes that have no fill kernel (the bitserial
// Bits* dtypes and Float4_e2m1fn_x2) and that it correctly zeroes a dense
// float tensor via the device memset.

#include <ATen/ATen.h>
#include <ATen/ops/zero_native.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>

#include <iostream>
#include <vector>

namespace {

// dtypes with no fill kernel on XPU (see intel/torch-xpu-ops#5378)
std::vector<at::ScalarType> kNoFillDtypes = {
    at::ScalarType::Bits1x8,
    at::ScalarType::Bits2x4,
    at::ScalarType::Bits4x2,
    at::ScalarType::Bits8,
    at::ScalarType::Bits16,
    at::ScalarType::Float4_e2m1fn_x2,
};

void test_zero_xpu_direct_clears_dtypes() {
  for (auto dtype : kNoFillDtypes) {
    auto raw = at::ones(
        {1024},
        at::TensorOptions().dtype(at::kByte)
            .device(at::Device(c10::kXPU, 0)));
    auto view = raw.view(dtype);
    at::native::zero_xpu_(view);
    TORCH_CHECK(
        raw.count_nonzero().item<int64_t>() == 0,
        "zero_xpu_ did not clear dtype ", toString(dtype));
  }
  std::cout << "PASS: zero_xpu_ clears bitserial/Float4 dtypes" << std::endl;
}

void test_zero_xpu_direct_memssets_dense() {
  auto t = at::full(
      {64, 64},
      1.0,
      at::TensorOptions().dtype(at::kFloat)
          .device(at::Device(c10::kXPU, 0)));
  at::native::zero_xpu_(t);
  TORCH_CHECK(
      t.count_nonzero().item<int64_t>() == 0,
      "zero_xpu_ did not zero a dense float tensor");
  std::cout << "PASS: zero_xpu_ zeroes a dense float tensor" << std::endl;
}

}  // namespace

int main() {
  test_zero_xpu_direct_clears_dtypes();
  test_zero_xpu_direct_memssets_dense();
  return 0;
}