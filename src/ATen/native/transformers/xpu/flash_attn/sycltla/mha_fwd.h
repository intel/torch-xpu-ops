/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once
#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/util/packed_stride.hpp>
#include <sycl/sycl.hpp>

#include "collective/xe_flash_attn_sdpa_fwd_epilogue.h"
#include "collective/xe_flash_attn_sdpa_fwd_mma.h"
#include "collective/xe_flash_attn_sdpa_fwd_softmax_epilogue.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "kernel/tile_scheduler_sdpa_fwd.h"
#include "kernel/xe_sdpa_fwd.h"