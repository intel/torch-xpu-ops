/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Explicit instantiations split across files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py".

#include <ATen/native/xpu/grouped_mm/sycltla/grouped_mm_kernel_impl.h>

namespace sycltla {
namespace grouped_gemm {

template void run_m_grouped<DType::FP16, 'R'>(sycl::queue&, const Params&);
template void run_m_grouped<DType::FP16, 'C'>(sycl::queue&, const Params&);
} // namespace grouped_gemm
} // namespace sycltla
