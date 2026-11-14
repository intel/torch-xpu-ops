/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/native/xpu/grouped_mm/sycltla/grouped_mm_kernels.h>

#include <algorithm>

namespace sycltla {
namespace grouped_gemm {

namespace {
bool is_pvc(const sycl::device& dev) {
  namespace ex = sycl::ext::oneapi::experimental;
  try {
    const auto arch = dev.get_info<ex::info::device::architecture>();
    return arch == ex::architecture::intel_gpu_pvc ||
        arch == ex::architecture::intel_gpu_pvc_vg;
  } catch (...) {
    return false;
  }
}

} // namespace

int xe_core_count(const sycl::device& dev) {
  int n = 0;
  try {
    n = int(dev.get_info<sycl::ext::intel::info::device::gpu_slices>()) *
        int(dev.get_info<
            sycl::ext::intel::info::device::gpu_subslices_per_slice>());
  } catch (...) {
    n = 0;
  }
  if (n <= 0) {
    // 8 EUs per Xe core on PVC/BMG.
    n = std::max(
        1, int(dev.get_info<sycl::info::device::max_compute_units>()) / 8);
  }
  return n;
}
int default_tile(const sycl::device& dev) {
  return is_pvc(dev) ? kTile256x256 : kTile256x128;
}

int choose_tile(const sycl::device& dev, const Params& p) {
  if (p.tile >= kTile256x128 && p.tile <= kTile128x256) {
    return p.tile;
  }
  // Only properties that are the same on every rank pick the tile: the
  // routed row counts differ between the ranks of an expert-parallel job,
  // and a rule on them would run different kernels for the same op. For
  // small groups (<= 128 rows per group on PVC, e.g. many experts or few
  // tokens) TORCH_XPU_GROUPED_MM_TILE=4 (128x256) is up to 42% faster
  // (tuner sweep, DSv3 16B shapes); it applies to all ranks alike.
  return default_tile(dev);
}

int default_num_workgroups(const sycl::device& dev, int tile) {
  // One 512-thread work-group fills an Xe core in GRF-256 mode; smaller
  // tiles launch proportionally more work-groups per core.
  const int threads = tile == kTile256x256 ? 512
      : tile == kTile128x128               ? 128
                                           : 256;
  return xe_core_count(dev) * (512 / threads);
}

const char* tile_name(int tile) {
  switch (tile) {
    case kTile256x128:
      return "256x128";
    case kTile256x256:
      return "256x256";
    case kTile128x128:
      return "128x128";
    case kTile128x256:
      return "128x256";
    default:
      return "auto";
  }
}

} // namespace grouped_gemm
} // namespace sycltla
