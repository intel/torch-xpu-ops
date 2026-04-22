/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*
 * BSD License
 * 
 * For FBGEMM software
 * 
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 *  * Neither the name Facebook nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

namespace fbgemm_utils {
    enum ArgIndex_aux_tensor {
        IDX_B_OFFSETS = 0,
        IDX_VBE_OUTPUT_OFFSETS_FEATURE_RANK = 1,
        IDX_VBE_B_OFFSETS_RANK_PER_FEATURE = 2,
        IDX_LXU_CACHE_LOCATIONS = 3,
        IDX_UVM_CACHE_STATS = 4,
        IDX_PREV_ITER_DEV = 5,
        AUX_TENSOR_SIZE = 10
    };

    enum ArgIndex_aux_bool {
        IDX_IS_EXPERIMENTAL_TBE = 0,
        IDX_USE_UNIQ_CACHE_LOCATIONS_BWD = 1,
        IDX_USE_HOMOGENEOUS_PLACEMENTS = 2,
        IDX_APPLY_GLOBAL_WEIGHT_DECAY = 3,
        IDX_GRADIENT_CLIPPING = 4,
        IDX_STOCHASTIC_ROUNDING = 5,
        IDX_MIXED_D = 6,
        AUX_BOOL_SIZE = 8
    };

    enum ArgIndex_aux_int {
        IDX_ITER = 0,
        IDX_INFO_B_NUM_BITS = 1,
        IDX_INFO_B_MASK = 2,
        AUX_INT_SIZE = 7
    };

    enum ArgIndex_aux_float {
        IDX_GWD_LOWER_BOUND = 0,
        IDX_MAX_GRADIENT = 1,
        AUX_FLOAT_SIZE = 9
    };
} // namespace fbgemm_utils
