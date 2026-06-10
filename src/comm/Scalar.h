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

#include <sycl/sycl.hpp>

#include <bit>

static inline uint32_t __float_as_int(float val) {
  return std::bit_cast<uint32_t>(val);
}

static inline float __int_as_float(uint32_t val) {
  return std::bit_cast<float>(val);
}

static inline unsigned long long __double_as_long_long(double val) {
  return std::bit_cast<unsigned long long>(val);
}

static inline double __long_long_as_double(unsigned long long val) {
  return std::bit_cast<double>(val);
}

static inline sycl::half __ushort_as_half(unsigned short int val) {
  return std::bit_cast<sycl::half>(val);
}
