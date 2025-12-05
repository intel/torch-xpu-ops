/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Create an idx array on SYCL GPU device
// res      - host buffer for result
// numel    - length of the idx array
void itoa(float* res, int numel);
