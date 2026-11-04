/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Tile-config candidate interface. tune_cfg.cpp, compiled once per config
// (-DTUNE_CFG_NAME=<name>), instantiates the production kernels for that
// config and exposes the three launchers below.

#pragma once

#include <sycl/sycl.hpp>

#include <functional>
#include <string>
#include <vector>

#include "grouped_mm_kernels.h"

namespace tune {

using sycltla::grouped_gemm::Params;
using Launch = std::function<void(sycl::queue&, const Params&, int num_wgs)>;

struct Candidate {
  std::string name;
  std::string desc;   // tile / sub-group layout / prefetch
  int threads = 0;    // work-group size
  Launch m_rowB;      // M-grouped, B (n, k) at n * ldb + k   (K-contiguous)
  Launch m_colB;      // M-grouped, B (n, k) at n + k * ldb   (N-contiguous)
  Launch k_grouped;   // K-grouped (grad_weight)
};

// Each config is built as its own shared object (so the AOT device compiles
// run in parallel and a config that fails to compile does not block the rest)
// exporting this entry point; the driver dlopen()s every libtune_cfg_*.so it
// finds.
using CandidateFn = const Candidate* (*)();
constexpr const char* kCandidateSymbol = "tune_grouped_mm_candidate";

} // namespace tune
