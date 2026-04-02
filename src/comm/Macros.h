/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#ifdef _WIN32
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

// Suppress -Wdeprecated-declarations warnings from oneAPI SYCL headers
// (files compiled without -fsycl). MSVC: C4996.
// Usage: wrap the offending #include(s) with BEGIN/END.
#if defined(_MSC_VER)
// clang-format off
#define DISABLE_SYCL_DEPRECATED_WARNING_BEGIN \
  __pragma(warning(push))                     \
  __pragma(warning(disable : 4996))
#define DISABLE_SYCL_DEPRECATED_WARNING_END __pragma(warning(pop))
// clang-format on
#elif defined(__GNUC__) || defined(__clang__)
// clang-format off
#define DISABLE_SYCL_DEPRECATED_WARNING_BEGIN                                   \
  _Pragma("GCC diagnostic push")                                                \
  _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#define DISABLE_SYCL_DEPRECATED_WARNING_END _Pragma("GCC diagnostic pop")
// clang-format on
#else
#define DISABLE_SYCL_DEPRECATED_WARNING_BEGIN
#define DISABLE_SYCL_DEPRECATED_WARNING_END
#endif
