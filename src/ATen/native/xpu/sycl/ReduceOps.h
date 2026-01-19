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

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

void argmax_kernel(TensorIterator& iter);

void and_kernel(TensorIterator& iter);

void or_kernel(TensorIterator& iter);

void mean_kernel(TensorIterator& iter);

void sum_kernel(TensorIterator& iter);

} // namespace at::native::xpu
