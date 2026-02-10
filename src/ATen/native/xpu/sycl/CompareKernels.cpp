/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/TensorIterator.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/Loops.h>

#include <ATen/native/xpu/sycl/CompareKernels.h>

namespace at {
namespace native {
namespace xpu {

enum class EqOpType { EQ, NE };

template <typename scalar_t>
struct CompareEqFunctor {
  CompareEqFunctor(EqOpType op) : op_(op) {}
  const EqOpType op_;
  bool operator()(scalar_t a, scalar_t b) const {
    if (op_ == EqOpType::EQ) {
      return a == b;
    } else { // NE
      return a != b;
    }
  }
};

void compare_eq_ne_kernel(TensorIteratorBase& iter, EqOpType op) {
  AT_DISPATCH_V2(
      iter.common_dtype(),
      "compare_eq_ne_xpu",
      AT_WRAP([&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t, bool>(
            iter, CompareEqFunctor<scalar_t>(op));
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      kComplexHalf,
      kHalf,
      kBFloat16,
      kBool,
      AT_EXPAND(AT_FLOAT8_TYPES),
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

void eq_kernel(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::EQ);
}

void ne_kernel(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::NE);
}

enum class OpType { GE, GT, LE, LT };

template <typename scalar_t>
struct CompareFunctor {
  constexpr CompareFunctor(OpType op) : op_(op){};
  OpType op_;
  bool operator()(scalar_t a, scalar_t b) const {
    if (op_ == OpType::GE) {
      return a >= b;
    } else if (op_ == OpType::GT) {
      return a > b;
    } else if (op_ == OpType::LE) {
      return a <= b;
    } else { // LT
      return a < b;
    }
  }
};

// Reflects the comparison operator, so reflect(op)(a, b) == op(b, a)
OpType reflect(OpType x) {
  switch (x) {
    case OpType::GE:
      return OpType::LE;
    case OpType::GT:
      return OpType::LT;
    case OpType::LE:
      return OpType::GE;
    case OpType::LT:
      return OpType::GT;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid OpType");
}

template <typename scalar_t, typename fn_t>
struct CompareScalarFunctor {
  CompareScalarFunctor(scalar_t rhs, fn_t fn) : rhs_(rhs), fn_(fn) {}
  bool operator()(scalar_t lhs) const {
    return fn_(lhs, rhs_);
  }
  scalar_t rhs_;
  fn_t fn_;
};

template <typename scalar_t>
void compare_scalar_kernel(TensorIteratorBase& iter, OpType op, scalar_t rhs) {
  CompareFunctor<scalar_t> f(op);
  auto caller = CompareScalarFunctor<scalar_t, decltype(f)>(rhs, f);
  gpu_kernel(iter, caller);
}

template <typename scalar_t>
void compare_kernel_impl(TensorIteratorBase& iter, OpType op) {
  // If either input is a cpu scalar, perform the equivalent comparison
  // where the scalar is on the right hand side. This saves us from
  // generating two otherwise identical kernels with mirrored
  // arguments.
  if (iter.is_cpu_scalar(1)) {
    const scalar_t lhs = iter.scalar_value<scalar_t>(1);
    iter.remove_operand(1);
    const DeviceGuard device_guard(iter.device(1));
    compare_scalar_kernel(iter, reflect(op), lhs);
  } else if (iter.is_cpu_scalar(2)) {
    const scalar_t rhs = iter.scalar_value<scalar_t>(2);
    iter.remove_operand(2);
    compare_scalar_kernel(iter, op, rhs);
  } else {
    CompareFunctor<scalar_t> f(op);
    gpu_kernel(iter, f);
  }
}

inline void compare_kernel_with_scalars(TensorIteratorBase& iter, OpType op) {
  AT_DISPATCH_V2(
      iter.common_dtype(),
      "compare_xpu",
      [&]() { compare_kernel_impl<scalar_t>(iter, op); },
      AT_EXPAND(AT_ALL_TYPES),
      kHalf,
      kBFloat16,
      kBool,
      AT_EXPAND(AT_FLOAT8_TYPES));
}

void ge_kernel(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::GE);
}

void gt_kernel(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::GT);
}

void le_kernel(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::LE);
}

void lt_kernel(TensorIteratorBase& iter) {
  compare_kernel_with_scalars(iter, OpType::LT);
}

} // namespace xpu
} // namespace native
} // namespace at
