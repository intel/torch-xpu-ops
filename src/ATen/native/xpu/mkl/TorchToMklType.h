/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/ATen.h>
#define SYCL_DISABLE_FSYCL_SYCLHPP_WARNING
#include <oneapi/mkl/blas.hpp>
#undef SYCL_DISABLE_FSYCL_SYCLHPP_WARNING

template <typename T>
struct get_mkl_type {
  using type = T;
};

template <typename T>
struct get_mkl_type<c10::complex<T>> {
  using type = std::complex<T>;
};

template <>
struct get_mkl_type<at::BFloat16> {
  using type = oneapi::mkl::bfloat16;
};

template <>
struct get_mkl_type<at::Half> {
  using type = sycl::half;
};
