/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/grouped_mm/grouped_mm_api.h>

#include <c10/util/Exception.h>
#include <c10/xpu/XPUStream.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_set>

#ifdef USE_SYCLTLA
#include <ATen/native/xpu/grouped_mm/sycltla/grouped_mm_kernels.h>
#endif

namespace sycltla {

bool is_grouped_mm_available() {
#ifdef USE_SYCLTLA
  return true;
#else
  return false;
#endif
}

#ifdef USE_SYCLTLA
namespace {

// Xe 2D block loads/stores need 64-byte aligned base addresses. Group bases
// are computed from arbitrary row offsets, so every leading dimension and
// group stride must itself be a multiple of 64 bytes (32 16-bit elements).
constexpr int64_t kLdAlignElems = 32;
// Surface widths must be a multiple of 16 bytes.
constexpr int64_t kDimAlignElems = 8;

bool is_aligned_64(const void* p) {
  return (reinterpret_cast<std::uintptr_t>(p) & 63u) == 0;
}

bool fits_int32(int64_t v) {
  return v >= 0 && v <= int64_t(std::numeric_limits<int32_t>::max());
}

struct Plan {
  bool k_grouped = false;
  char layout_b = 'C';
  grouped_gemm::DType dtype = grouped_gemm::DType::BF16;
  grouped_gemm::Params params{};
};

// Fills `plan`; returns an empty string on success, otherwise the reason the
// SYCL-TLA path cannot be used.
std::string build_plan(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    const at::Tensor& out,
    Plan& plan) {
  if (!mat_a.defined() || !mat_b.defined() || !offs.defined() ||
      !out.defined()) {
    return "undefined tensor";
  }
  if (!mat_a.is_xpu() || !mat_b.is_xpu() || !offs.is_xpu() || !out.is_xpu()) {
    return "all tensors must be on XPU";
  }
  if (mat_a.layout() != at::kStrided || mat_b.layout() != at::kStrided ||
      out.layout() != at::kStrided) {
    return "only strided tensors";
  }
  const auto dtype = mat_a.scalar_type();
  if (dtype != at::kBFloat16 && dtype != at::kHalf) {
    return "only bf16/fp16";
  }
  if (mat_b.scalar_type() != dtype || out.scalar_type() != dtype) {
    return "mat_a, mat_b and out must share a dtype";
  }
  if (offs.dim() != 1 || offs.scalar_type() != at::kInt ||
      !offs.is_contiguous() || offs.numel() == 0) {
    return "offs must be a non-empty contiguous int32 1-D tensor";
  }
  plan.dtype = dtype == at::kBFloat16 ? grouped_gemm::DType::BF16
                                      : grouped_gemm::DType::FP16;

  auto& p = plan.params;
  p.a = mat_a.const_data_ptr();
  p.b = mat_b.const_data_ptr();
  p.d = out.data_ptr();
  p.offs = offs.const_data_ptr<int32_t>();
  p.num_groups = int32_t(offs.numel());
  if (!is_aligned_64(p.a) || !is_aligned_64(p.b) || !is_aligned_64(p.d)) {
    return "base pointers must be 64-byte aligned";
  }

  if (mat_a.dim() == 2 && mat_b.dim() == 3) {
    // D[rows_g, N] = A[rows_g, K] @ B[g][K, N]
    plan.k_grouped = false;
    const int64_t M = mat_a.size(0), K = mat_a.size(1);
    const int64_t E = mat_b.size(0), N = mat_b.size(2);
    if (mat_b.size(1) != K) {
      return "contraction dims differ";
    }
    if (E != offs.numel()) {
      return "mat_b.size(0) must equal offs.numel()";
    }
    if (out.dim() != 2 || out.size(0) != M || out.size(1) != N) {
      return "unexpected out shape";
    }
    if (!fits_int32(M) || !fits_int32(N) || !fits_int32(K)) {
      return "shape exceeds int32";
    }
    if (mat_a.stride(1) != 1) {
      return "mat_a must be K-contiguous (row-major)";
    }
    if (out.stride(1) != 1) {
      return "out must be N-contiguous (row-major)";
    }
    const int64_t s_k = mat_b.stride(1), s_n = mat_b.stride(2);
    if (s_n == 1) {
      plan.layout_b = 'C'; // row-major [K, N] weight
      p.ldb = s_k;
    } else if (s_k == 1) {
      plan.layout_b = 'R'; // transposed weight, K-contiguous
      p.ldb = s_n;
    } else {
      return "mat_b must be K- or N-contiguous per group";
    }
    p.lda = mat_a.stride(0);
    p.ldd = out.stride(0);
    p.group_stride_b = mat_b.stride(0);
    p.M = int32_t(M);
    p.N = int32_t(N);
    p.K = int32_t(K);
    if (K % kDimAlignElems != 0 || N % kDimAlignElems != 0) {
      return "K and N must be multiples of 8 elements";
    }
    if (p.lda % kLdAlignElems != 0 || p.ldb % kLdAlignElems != 0 ||
        p.ldd % kLdAlignElems != 0 || p.group_stride_b % kLdAlignElems != 0) {
      return "leading dimensions must be multiples of 64 bytes";
    }
    return "";
  }

  if (mat_a.dim() == 2 && mat_b.dim() == 2) {
    // D[g][M, N] = A[M, rows_g] @ B[rows_g, N]
    plan.k_grouped = true;
    const int64_t M = mat_a.size(0), R = mat_a.size(1);
    const int64_t N = mat_b.size(1);
    if (mat_b.size(0) != R) {
      return "contraction dims differ";
    }
    if (out.dim() != 3 || out.size(0) != offs.numel() || out.size(1) != M ||
        out.size(2) != N) {
      return "unexpected out shape";
    }
    if (!fits_int32(M) || !fits_int32(N) || !fits_int32(R)) {
      return "shape exceeds int32";
    }
    if (mat_a.stride(0) != 1) {
      return "mat_a must be M-contiguous (transposed activation)";
    }
    if (mat_b.stride(1) != 1) {
      return "mat_b must be N-contiguous (row-major)";
    }
    if (out.stride(2) != 1) {
      return "out must be N-contiguous (row-major)";
    }
    p.lda = mat_a.stride(1);
    p.ldb = mat_b.stride(0);
    p.ldd = out.stride(1);
    p.group_stride_d = out.stride(0);
    p.M = int32_t(M);
    p.N = int32_t(N);
    p.K = int32_t(R);
    if (M % kDimAlignElems != 0 || N % kDimAlignElems != 0) {
      return "M and N must be multiples of 8 elements";
    }
    if (p.lda % kLdAlignElems != 0 || p.ldb % kLdAlignElems != 0 ||
        p.ldd % kLdAlignElems != 0 || p.group_stride_d % kLdAlignElems != 0) {
      return "leading dimensions must be multiples of 64 bytes";
    }
    return "";
  }

  return "only 2D x 3D and 2D x 2D grouped GEMMs are supported";
}

// Tuning knob: force the tile config (1 = 256x128, 2 = 256x256,
// 3 = 128x128, 4 = 128x256); 0/unset => grouped_gemm::choose_tile().
int32_t tile_from_env() {
  const char* v = std::getenv("TORCH_XPU_GROUPED_MM_TILE");
  if (v == nullptr || v[0] == '\0') {
    return 0;
  }
  const long t = std::strtol(v, nullptr, 10);
  return (t >= 1 && t <= 4) ? int32_t(t) : 0;
}

int32_t num_workgroups_from_env() {
  const char* v = std::getenv("TORCH_XPU_GROUPED_MM_NUM_WORKGROUPS");
  if (v == nullptr || v[0] == '\0') {
    return 0;
  }
  const long n = std::strtol(v, nullptr, 10);
  return n > 0 ? int32_t(n) : 0;
}

void log_plan(
    const Plan& plan,
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& out) {
  const auto& p = plan.params;
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  grouped_gemm::Params q = p;
  q.tile = tile_from_env();
  const int tile = grouped_gemm::choose_tile(queue.get_device(), q);
  const int wgs = num_workgroups_from_env() > 0
      ? num_workgroups_from_env()
      : grouped_gemm::default_num_workgroups(queue.get_device(), tile);
  const char* dt = plan.dtype == grouped_gemm::DType::BF16 ? "bf16" : "fp16";
  const char* tile_s = grouped_gemm::tile_name(tile);
  std::string kind = plan.k_grouped ? std::string("k-grouped(2Dx2D)")
                                    : std::string("m-grouped(2Dx3D) B=") +
          (plan.layout_b == 'C' ? "N-contig" : "K-contig");
  const bool empty =
      out.numel() == 0 || mat_a.numel() == 0 || mat_b.numel() == 0;
  std::string key = std::string("sycltla|") + kind + "|" + dt + "|" +
      std::to_string(p.N) + "|" + std::to_string(plan.k_grouped ? p.M : p.K) +
      "|" + std::to_string(p.num_groups) + (empty ? "|empty" : "");
  std::string line = c10::str(
      "SYCL-TLA ",
      kind,
      " ",
      dt,
      empty ? " [empty: zero-fill, no launch]" : "",
      " tile=",
      tile_s,
      " wgs=",
      wgs,
      " groups=",
      p.num_groups,
      " a",
      mat_a.sizes(),
      " b",
      mat_b.sizes(),
      " -> out",
      out.sizes());
  log_grouped_mm(key, line);
}

} // namespace
#endif // USE_SYCLTLA

int grouped_mm_log_level() {
  const char* v = std::getenv("TORCH_XPU_GROUPED_MM_LOG");
  if (v == nullptr || v[0] == '\0') {
    return 0;
  }
  return int(std::strtol(v, nullptr, 10));
}

void log_grouped_mm(const std::string& key, const std::string& line) {
  const int level = grouped_mm_log_level();
  if (level <= 0) {
    return;
  }
  static std::mutex mu;
  static std::unordered_set<std::string> seen;
  std::lock_guard<std::mutex> guard(mu);
  if (level == 1 && !seen.insert(key).second) {
    return;
  }
  std::fprintf(stderr, "[xpu grouped_mm] %s\n", line.c_str());
  std::fflush(stderr);
}

bool can_use_grouped_mm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    const at::Tensor& out,
    std::string* why_not) {
#ifndef USE_SYCLTLA
  if (why_not != nullptr) {
    *why_not = "torch-xpu-ops built without SYCL-TLA";
  }
  return false;
#else
  Plan plan;
  std::string reason = build_plan(mat_a, mat_b, offs, out, plan);
  if (why_not != nullptr) {
    *why_not = reason;
  }
  return reason.empty();
#endif
}

void grouped_mm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out) {
#ifndef USE_SYCLTLA
  TORCH_CHECK(
      false, "sycltla::grouped_mm: torch-xpu-ops was built without SYCL-TLA");
#else
  Plan plan;
  const std::string reason = build_plan(mat_a, mat_b, offs, out, plan);
  TORCH_CHECK(
      reason.empty(), "sycltla::grouped_mm: unsupported inputs: ", reason);
  if (grouped_mm_log_level() > 0) {
    log_plan(plan, mat_a, mat_b, out);
  }
  if (out.numel() == 0) {
    return;
  }
  if (mat_a.numel() == 0 || mat_b.numel() == 0) {
    // Empty contraction: every group is empty along K (2D x 2D with no rows,
    // or K == 0). The result is all zeros. Do not launch: data_ptr() of an
    // empty tensor is nullptr and the kernel would dereference it.
    out.zero_();
    return;
  }
  plan.params.num_workgroups = num_workgroups_from_env();
  plan.params.tile = tile_from_env();

  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  using grouped_gemm::DType;
  if (plan.k_grouped) {
    if (plan.dtype == DType::BF16) {
      grouped_gemm::run_k_grouped<DType::BF16>(queue, plan.params);
    } else {
      grouped_gemm::run_k_grouped<DType::FP16>(queue, plan.params);
    }
    return;
  }
  if (plan.dtype == DType::BF16) {
    if (plan.layout_b == 'C') {
      grouped_gemm::run_m_grouped<DType::BF16, 'C'>(queue, plan.params);
    } else {
      grouped_gemm::run_m_grouped<DType::BF16, 'R'>(queue, plan.params);
    }
  } else {
    if (plan.layout_b == 'C') {
      grouped_gemm::run_m_grouped<DType::FP16, 'C'>(queue, plan.params);
    } else {
      grouped_gemm::run_m_grouped<DType::FP16, 'R'>(queue, plan.params);
    }
  }
#endif
}

} // namespace sycltla
