/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef _WIN32
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif
