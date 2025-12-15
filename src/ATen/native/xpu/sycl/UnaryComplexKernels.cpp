/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <comm/xpu_aten.h>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/ScalarType.h>

#include <ATen/native/xpu/sycl/CopyKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/CopyKernel.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/UnaryComplexKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct ConjScalarFunc {
  scalar_t operator()(scalar_t src_val) const {
    return std::conj(src_val);
  }
};

void conj_kernel(TensorIterator& iter) {
  AT_DISPATCH_SWITCH(
      iter.common_dtype(),
      "conj_xpu",
      AT_DISPATCH_CASE_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, [&] {
        // Conj is a no-op for non-complex types
        copy_kernel(iter);
      }) AT_DISPATCH_CASE_COMPLEX_TYPES_AND(kComplexHalf, [&] {
        gpu_kernel(iter, ConjScalarFunc<scalar_t>());
      }));
}

template <typename scalar_t>
struct ConjPhysicalFunctor {
  scalar_t operator()(scalar_t z) const {
    return std::conj(z);
  }
};

template <typename TYPE>
struct ConjPhysicalFunctor<c10::complex<TYPE>> {
  c10::complex<TYPE> operator()(c10::complex<TYPE> z) const {
    return c10::complex<TYPE>(z.real(), -z.imag());
  }
};

void conj_physical_kernel(TensorIteratorBase& iter) {
  AT_DISPATCH_SWITCH(
      iter.common_dtype(),
      "conj_xpu",
      AT_DISPATCH_CASE_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, [&] {
        // Conj is a no-op for non-complex types
        copy_kernel(iter);
      }) AT_DISPATCH_CASE_COMPLEX_TYPES_AND(kComplexHalf, [&] {
        gpu_kernel(iter, ConjPhysicalFunctor<scalar_t>());
      }));
}

template <typename scalar_t>
struct NegConjScalarFunc {
  scalar_t operator()(scalar_t src_val) const {
    return std::conj(-src_val);
  }
};

void neg_conj_kernel(TensorIterator& iter) {
  AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "neg_conj_xpu", [&] {
    gpu_kernel(iter, NegConjScalarFunc<scalar_t>());
  });
}

template <typename scalar_t>
struct NegScalarFunc {
  scalar_t operator()(scalar_t src_val) const {
    return -src_val;
  }
};

void neg_kernel(TensorIterator& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "neg_xpu", [&]() {
      gpu_kernel(iter, NegScalarFunc<scalar_t>());
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(
        ScalarType::Half, ScalarType::BFloat16, dtype, "neg_xpu", [&]() {
          gpu_kernel(iter, NegScalarFunc<scalar_t>());
        });
  }
}

template <typename scalar_t>
struct AngleWrapper {
  scalar_t operator()(scalar_t v) const {
    if (at::_isnan(v)) {
      return v;
    }
    return v < 0 ? M_PI : 0;
  }
};

template <typename T>
struct AngleWrapper<c10::complex<T>> {
  c10::complex<T> operator()(c10::complex<T> v) const {
    return c10::complex<T>{std::arg(v), 0};
  }
};

void angle_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "angle_xpu", [&]() {
      gpu_kernel(iter, AngleWrapper<scalar_t>());
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES(dtype, "angle_xpu", [&]() {
      gpu_kernel(iter, AngleWrapper<scalar_t>());
    });
  }
}

} // namespace at::native::xpu
