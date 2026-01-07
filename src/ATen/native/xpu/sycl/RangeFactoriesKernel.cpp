/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/FunctionTraits.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/RangeFactoriesKernel.h>

namespace at::native::xpu {

constexpr int nitem_per_wg = 256;
constexpr int item_work_size = 1;
constexpr int group_work_size = item_work_size * nitem_per_wg;

template <typename index_t, typename func_t>
struct ElementwiseKernelWithIndexFunctor {
  using res_t = typename function_traits<func_t>::result_type;
  void operator()(sycl::nd_item<1> item) const {
#pragma unroll
    for (int i = 0; i < item_work_size; i++) {
      index_t idx = group_work_size * item.get_group(0) + nitem_per_wg * i +
          item.get_local_id(0);
      if (idx < N_) {
        data_[idx] = f_(idx);
      }
    }
  }
  ElementwiseKernelWithIndexFunctor(index_t N, func_t f, res_t* data)
      : N_(N), f_(f), data_(data) {}

 private:
  index_t N_;
  func_t f_;
  res_t* data_;
};

template <typename func_t>
void gpu_kernel_with_index(at::Tensor& output, func_t f) {
  int64_t N = output.numel();
  if (N == 0) {
    return;
  }
  int64_t num_wg = (N + group_work_size - 1) / group_work_size;
  auto queue = at::xpu::getCurrentSYCLQueue();
  using scalar_t = typename function_traits<func_t>::result_type;
  if (N <= std::numeric_limits<int>::max()) {
    auto caller = ElementwiseKernelWithIndexFunctor<int, func_t>(
        N, f, output.mutable_data_ptr<scalar_t>());
    sycl_kernel_submit(num_wg * nitem_per_wg, nitem_per_wg, queue, caller);
  } else {
    auto caller = ElementwiseKernelWithIndexFunctor<int64_t, func_t>(
        N, f, output.mutable_data_ptr<scalar_t>());
    sycl_kernel_submit(num_wg * nitem_per_wg, nitem_per_wg, queue, caller);
  }
}

template <typename scalar_t, typename accscalar_t>
struct ArangeFunctor {
  scalar_t operator()(int64_t ind) const {
    accscalar_t inc = xstep_ * static_cast<accscalar_t>(ind);
    accscalar_t val = xstart_ + inc;
    return static_cast<scalar_t>(val);
  }
  ArangeFunctor(accscalar_t xstart, accscalar_t xstep)
      : xstart_(xstart), xstep_(xstep) {}

 private:
  accscalar_t xstart_;
  accscalar_t xstep_;
};

Tensor& arange_kernel(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      result.scalar_type(),
      "arange_xpu",
      [&]() {
        using accscalar_t = at::acc_type_device<scalar_t, kXPU>;
        auto xstart = start.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        bool is_contiguous = result.is_contiguous();
        Tensor r = !is_contiguous
            ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
            : result;

        auto f = ArangeFunctor<scalar_t, accscalar_t>(xstart, xstep);
        gpu_kernel_with_index(r, f);

        if (!is_contiguous) {
          result.copy_(r);
        }
      });

  return result;
}

template <typename scalar_t, typename accscalar_t>
struct RangeFunctor {
  scalar_t operator()(int64_t ind) const {
    accscalar_t inc = xstep_ * static_cast<accscalar_t>(ind);
    accscalar_t val = xstart_ + inc;
    return static_cast<scalar_t>(val);
  }
  RangeFunctor(accscalar_t xstart, accscalar_t xstep)
      : xstart_(xstart), xstep_(xstep) {}

 private:
  accscalar_t xstart_;
  accscalar_t xstep_;
};

Tensor& range_kernel(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& result) {
  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::Half, result.scalar_type(), "range_xpu", [&]() {
        using accscalar_t = acc_type_device<scalar_t, kXPU>;
        auto xstart = start.to<accscalar_t>();
        auto xstep = step.to<accscalar_t>();

        bool is_contiguous = result.is_contiguous();
        Tensor r = !is_contiguous
            ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
            : result;
        auto f = RangeFunctor<scalar_t, accscalar_t>(xstart, xstep);

        gpu_kernel_with_index(r, f);

        if (!result.is_contiguous()) {
          result.copy_(r);
        }
      });

  return result;
}

template <typename scalar_t, typename step_type>
struct LinspaceFunctor {
  scalar_t operator()(int64_t ind) const {
    if (ind < halfway_) {
      return scalar_start_ + (step_ * ind);
    }

    return scalar_end_ - step_ * (steps_ - ind - 1);
  }
  LinspaceFunctor(
      scalar_t scalar_start,
      scalar_t scalar_end,
      int64_t steps,
      step_type step,
      int64_t halfway)
      : scalar_start_(scalar_start),
        scalar_end_(scalar_end),
        steps_(steps),
        step_(step),
        halfway_(halfway) {}

 private:
  scalar_t scalar_start_;
  scalar_t scalar_end_;
  int64_t steps_;
  step_type step_;
  int64_t halfway_;
};

Tensor& linspace_kernel(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous
      ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
      : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    r.fill_(start);
  } else if (isIntegralType(r.scalar_type(), 0)) {
    AT_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "linspace_xpu", [&]() {
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      // Cast `end` and `start` to `float`, since range can be larger than
      // scalar_t for integral types
      float step =
          (static_cast<float>(scalar_end) - static_cast<float>(scalar_start)) /
          (steps - 1);
      const int64_t halfway = steps / 2;
      auto f = LinspaceFunctor<scalar_t, float>(
          scalar_start, scalar_end, steps, step, halfway);

      gpu_kernel_with_index(r, f);
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, r.scalar_type(), "linspace_xpu", [&]() {
          scalar_t scalar_start = start.to<scalar_t>();
          scalar_t scalar_end = end.to<scalar_t>();
          scalar_t step =
              (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
          const int64_t halfway = steps / 2;
          auto f = LinspaceFunctor<scalar_t, scalar_t>(
              scalar_start, scalar_end, steps, step, halfway);

          gpu_kernel_with_index(r, f);
        });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}

template <typename scalar_t, typename step_type>
struct LogspaceFunctor {
  scalar_t operator()(int64_t ind) const {
    if (ind < halfway_) {
      return std::pow(scalar_base_, scalar_start_ + step_ * ind);
    }

    return std::pow(scalar_base_, scalar_end_ - step_ * (steps_ - ind - 1));
  }
  LogspaceFunctor(
      scalar_t scalar_start,
      scalar_t scalar_end,
      step_type scalar_base,
      int64_t steps,
      step_type step,
      int64_t halfway)
      : scalar_start_(scalar_start),
        scalar_end_(scalar_end),
        scalar_base_(scalar_base),
        steps_(steps),
        step_(step),
        halfway_(halfway) {}

 private:
  scalar_t scalar_start_;
  scalar_t scalar_end_;
  step_type scalar_base_;
  int64_t steps_;
  step_type step_;
  int64_t halfway_;
};

Tensor& logspace_kernel(
    const Scalar& start,
    const Scalar& end,
    int64_t steps,
    double base,
    Tensor& result) {
  TORCH_CHECK(steps >= 0, "number of steps must be non-negative");

  if (result.numel() != steps) {
    result.resize_({steps});
  }
  bool is_contiguous = result.is_contiguous();
  Tensor r = !is_contiguous
      ? at::empty_like(result, LEGACY_CONTIGUOUS_MEMORY_FORMAT)
      : result;

  if (steps == 0) {
    // skip
  } else if (steps == 1) {
    if (isComplexType(r.scalar_type())) {
      r.fill_(std::pow(base, start.to<c10::complex<double>>()));
    } else {
      r.fill_(std::pow(base, start.to<double>()));
    }
  } else if (isIntegralType(r.scalar_type(), 0)) {
    AT_DISPATCH_INTEGRAL_TYPES(r.scalar_type(), "logspace_xpu", [&]() {
      float scalar_base =
          static_cast<float>(base); // Use float to avoid promotion to double
      scalar_t scalar_start = start.to<scalar_t>();
      scalar_t scalar_end = end.to<scalar_t>();
      float step = static_cast<float>(scalar_end - scalar_start) / (steps - 1);
      const int64_t halfway = steps / 2;
      auto f = LogspaceFunctor<scalar_t, float>(
          scalar_start, scalar_end, scalar_base, steps, step, halfway);

      gpu_kernel_with_index(r, f);
    });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
        kHalf, kBFloat16, r.scalar_type(), "logspace_xpu", [&]() {
          scalar_t scalar_base = static_cast<scalar_t>(base);
          scalar_t scalar_start = start.to<scalar_t>();
          scalar_t scalar_end = end.to<scalar_t>();
          scalar_t step =
              (scalar_end - scalar_start) / static_cast<scalar_t>(steps - 1);
          const int64_t halfway = steps / 2;
          auto f = LogspaceFunctor<scalar_t, scalar_t>(
              scalar_start, scalar_end, scalar_base, steps, step, halfway);

          gpu_kernel_with_index(r, f);
        });
  }

  if (!is_contiguous) {
    result.copy_(r);
  }

  return result;
}
} // namespace at::native::xpu
