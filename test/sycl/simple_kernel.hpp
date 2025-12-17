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

// Create an idx array on SYCL GPU device
// res      - host buffer for result
// numel    - length of the idx array
void itoa(float* res, int numel);
