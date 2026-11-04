/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>

#include <string>

namespace sycltla {

// True when torch-xpu-ops was built with SYCL-TLA (USE_SYCLTLA).
bool is_grouped_mm_available();

// Whether the SYCL-TLA grouped GEMM can serve this at::_grouped_mm call.
// Supported today (bf16 / fp16, same dtype for all operands, int32 offs):
//   * mat_a [M, K] row-major, mat_b [E, K, N] (K- or N-contiguous per group),
//     out [M, N] row-major  -> groups partition rows of mat_a / out.
//   * mat_a [K, M] M-contiguous (a transposed row-major activation), mat_b
//     [M, N] row-major, out [E, K, N] -> groups partition the contraction dim.
// Leading dimensions must be multiples of 64 bytes and base pointers 64-byte
// aligned (Xe 2D block message requirements). Anything else returns false so
// the caller can fall back.
// When it returns false and `why_not` is non-null, *why_not gets a short
// human-readable reason (used for dispatch logging).
bool can_use_grouped_mm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    const at::Tensor& out,
    std::string* why_not = nullptr);

// Tuning knobs read on every call: TORCH_XPU_GROUPED_MM_TILE (1 = 256x128,
// 2 = 256x256, 3 = 128x128, 4 = 128x256; unset = device default, see
// choose_tile()) and
// TORCH_XPU_GROUPED_MM_NUM_WORKGROUPS (persistent grid size).
// Backend selection (SYCL-TLA vs oneDNN vs fallback) is the caller's job; in
// PyTorch see TORCH_XPU_GROUPED_MM_BACKEND in
// aten/src/ATen/native/xpu/GroupedBlas.cpp.
//
// Runs the grouped GEMM on the current XPU stream. Requires
// can_use_grouped_mm() to hold; `out` is fully overwritten.
void grouped_mm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out);

// Dispatch logging, controlled by TORCH_XPU_GROUPED_MM_LOG (read on every
// call, so it can be toggled at runtime, e.g. via os.environ in Python):
//   unset/0  off
//   1        print each distinct dispatch decision once per process
//            (decisions are keyed on path, dtype, layouts and the fixed GEMM
//            dims, not on the varying row count), to stderr
//   2        print every call
int grouped_mm_log_level();

// Prints "[xpu grouped_mm] <line>" to stderr if logging is enabled; with
// level 1 only the first line for a given `key` is printed.
void log_grouped_mm(const std::string& key, const std::string& line);

} // namespace sycltla
