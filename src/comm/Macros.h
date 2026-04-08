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

// Suppress "not all control paths return a value" warning in SYCL kernel files.
// MSVC: C4715, GCC/Clang: -Wreturn-type
#if defined(_MSC_VER)
// clang-format off
#define DISABLE_RETURN_TYPE_WARNING_BEGIN \
  __pragma(warning(push))                \
  __pragma(warning(disable : 4715))
#define DISABLE_RETURN_TYPE_WARNING_END __pragma(warning(pop))
// clang-format on
#elif defined(__clang__)
// clang-format off
#define DISABLE_RETURN_TYPE_WARNING_BEGIN                              \
  _Pragma("clang diagnostic push")                                     \
  _Pragma("clang diagnostic ignored \"-Wreturn-type\"")
#define DISABLE_RETURN_TYPE_WARNING_END _Pragma("clang diagnostic pop")
// clang-format on
#elif defined(__GNUC__)
// clang-format off
#define DISABLE_RETURN_TYPE_WARNING_BEGIN                          \
  _Pragma("GCC diagnostic push")                                   \
  _Pragma("GCC diagnostic ignored \"-Wreturn-type\"")
#define DISABLE_RETURN_TYPE_WARNING_END _Pragma("GCC diagnostic pop")
// clang-format on
#else
#define DISABLE_RETURN_TYPE_WARNING_BEGIN
#define DISABLE_RETURN_TYPE_WARNING_END
#endif
