/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Candidate tile configs. Add a struct here and a name to CONFIGS in
// build.sh to tune it. Every config keeps the Cfg contract documented in
// grouped_mm_kernel_impl.h (WGTile, SGLayout, kTileM/N/K, kPrefetchDist).
// p256x128, p256x256, p128x128 and p128x256 are the four production configs.

#pragma once

#include "grouped_mm_kernel_impl.h"

namespace tune {
using namespace cute;

#define TUNE_CFG(NAME, M, N, K, SGM, SGN, PF)                                 \
  struct NAME {                                                               \
    using WGTile = Shape<Int<M>, Int<N>, Int<K>>;                             \
    using SGLayout = Layout<Shape<Int<SGM>, Int<SGN>, _1>,                     \
                            Stride<Int<SGN>, _1, _0>>;                        \
    static constexpr int kTileM = M;                                          \
    static constexpr int kTileN = N;                                          \
    static constexpr int kTileK = K;                                          \
    static constexpr int kPrefetchDist = PF;                                  \
    static constexpr const char* kDesc =                                      \
        #M "x" #N "x" #K " sg" #SGM "x" #SGN " pf" #PF;                       \
  };

//        name          tileM tileN tileK sgM sgN prefetch
TUNE_CFG(p256x128,      256,  128,  32,   8,  2,  3)   // production tile 1
TUNE_CFG(p256x256,      256,  256,  32,   8,  4,  3)   // production tile 2 (PVC)
TUNE_CFG(c256x256_pf2,  256,  256,  32,   8,  4,  2)
TUNE_CFG(c256x256_pf4,  256,  256,  32,   8,  4,  4)
TUNE_CFG(c256x256_k64,  256,  256,  64,   8,  4,  2)
TUNE_CFG(c256x128_k64,  256,  128,  64,   8,  2,  3)
TUNE_CFG(p128x256,      128,  256,  32,   4,  4,  3)   // production tile 4
TUNE_CFG(p128x128,      128,  128,  32,   4,  2,  3)   // production tile 3
TUNE_CFG(c512x128,      512,  128,  32,  16,  2,  3)
TUNE_CFG(c256x256_sg4x8, 256, 256,  32,   4,  8,  3)

#undef TUNE_CFG

} // namespace tune
