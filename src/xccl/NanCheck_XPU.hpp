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

#ifdef USE_C10D_XCCL

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

namespace c10d {

void checkForNan(const at::Tensor& tensor, at::xpu::XPUStream& stream);

} // namespace c10d

#endif // USE_C10D_XCCL
