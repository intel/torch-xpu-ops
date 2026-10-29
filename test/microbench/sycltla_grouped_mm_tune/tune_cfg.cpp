/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// One translation unit per tile config: compile with -DTUNE_CFG_NAME=<name>.

#include "tune_configs.h"
#include "tune_registry.h"

#ifndef TUNE_CFG_NAME
#error "define TUNE_CFG_NAME to one of the configs in tune_configs.h"
#endif
#define TUNE_STR2(x) #x
#define TUNE_STR(x) TUNE_STR2(x)

namespace {
using namespace sycltla::grouped_gemm;
using Cfg = tune::TUNE_CFG_NAME;
using T = cute::bfloat16_t;

const tune::Candidate kCandidate{
    TUNE_STR(TUNE_CFG_NAME),
    Cfg::kDesc,
    int(cute::size(TiledMmaFor<T, Cfg>{})),
    [](sycl::queue& q, const Params& p, int w) {
      run_m_grouped_cfg<Cfg, DType::BF16, 'R'>(q, p, w);
    },
    [](sycl::queue& q, const Params& p, int w) {
      run_m_grouped_cfg<Cfg, DType::BF16, 'C'>(q, p, w);
    },
    [](sycl::queue& q, const Params& p, int w) {
      run_k_grouped_cfg<Cfg, DType::BF16>(q, p, w);
    },
};
} // namespace

extern "C" const tune::Candidate* tune_grouped_mm_candidate() {
  return &kCandidate;
}
