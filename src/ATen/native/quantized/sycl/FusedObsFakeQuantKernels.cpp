/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/aminmax.h>
#endif

#include <ATen/native/quantized/sycl/FusedObsFakeQuantKernels.h>
#include <ATen/native/xpu/sycl/DistributionTemplates.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t>
void MovingAverageMinMax(
    const int64_t* observer_on,
    const scalar_t* x_min,
    const scalar_t* x_max,
    scalar_t* running_min,
    scalar_t* running_max,
    const float averaging_const,
    const int size,
    sycl::nd_item<1>& item) {
  int i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

  if (*observer_on == 1) {
    if (i < size) {
      scalar_t curr_min = x_min[i];
      scalar_t curr_max = x_max[i];

      scalar_t averaging_const_t = static_cast<scalar_t>(averaging_const);

      scalar_t adjusted_min =
          std::isinf(static_cast<at::opmath_type<scalar_t>>(running_min[i]))
          ? curr_min
          : (running_min[i]) +
              averaging_const_t * (curr_min - (running_min[i]));

      scalar_t adjusted_max =
          std::isinf(static_cast<at::opmath_type<scalar_t>>(running_max[i]))
          ? curr_max
          : (running_max[i]) +
              averaging_const_t * (curr_max - (running_max[i]));

      running_min[i] = adjusted_min;
      running_max[i] = adjusted_max;
    }
  }
}

template <typename scalar_t>
struct CalculateMovingAverageKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    MovingAverageMinMax<scalar_t>(
        observer_on_data_,
        x_min_data_,
        x_max_data_,
        running_min_data_,
        running_max_data_,
        averaging_const_,
        size_,
        item);
  }
  CalculateMovingAverageKernelFunctor(
      const int64_t* observer_on_data,
      const scalar_t* x_min_data,
      const scalar_t* x_max_data,
      scalar_t* running_min_data,
      scalar_t* running_max_data,
      const float averaging_const,
      const int64_t size)
      : observer_on_data_(observer_on_data),
        x_min_data_(x_min_data),
        x_max_data_(x_max_data),
        running_min_data_(running_min_data),
        running_max_data_(running_max_data),
        averaging_const_(averaging_const),
        size_(size) {}

 private:
  const int64_t* observer_on_data_;
  const scalar_t* x_min_data_;
  const scalar_t* x_max_data_;
  scalar_t* running_min_data_;
  scalar_t* running_max_data_;
  const float averaging_const_;
  const int64_t size_;
};

void _calculate_moving_average(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    const float averaging_const,
    const int64_t size,
    bool per_row_fake_quant) {
  auto execution_policy = calc_execution_policy(size);
  // auto counter_offset = std::get<0>(execution_policy);
  auto num_groups = std::get<1>(execution_policy);
  auto group_size = std::get<2>(execution_policy);

  at::Tensor x_min, x_max;

  int64_t* observer_on_data = observer_on.data_ptr<int64_t>();

  auto local_range = per_row_fake_quant ? group_size : 1;
  if (per_row_fake_quant) {
    std::tie(x_min, x_max) = at::aminmax(x, 1);
  } else {
    std::tie(x_min, x_max) = at::aminmax(x);
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, x.scalar_type(), "MovingAverageMinMax", [&] {
        scalar_t* x_min_data = x_min.data_ptr<scalar_t>();
        scalar_t* x_max_data = x_max.data_ptr<scalar_t>();
        scalar_t* running_min_data = running_min.data_ptr<scalar_t>();
        scalar_t* running_max_data = running_max.data_ptr<scalar_t>();

        // Moving Average Min/Max observer for activations
        CalculateMovingAverageKernelFunctor<scalar_t> kfn(
            observer_on_data,
            x_min_data,
            x_max_data,
            running_min_data,
            running_max_data,
            averaging_const,
            size);
        sycl_kernel_submit(
            num_groups * group_size, local_range, getCurrentSYCLQueue(), kfn);
      });
}

template <typename scalar_t>
void ChooseQuantizationParamsKernelImpl(
    const int64_t* fake_quant_on,
    const scalar_t* x_min,
    const scalar_t* x_max,
    int32_t qmin,
    int32_t qmax,
    int size,
    bool preserve_sparsity,
    float* scale,
    int32_t* zero_point,
    sycl::nd_item<1>& item) {
  int i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

  if (i < size && *fake_quant_on == 1) {
    float min_val = x_min[i];
    float max_val = x_max[i];

    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      int symmetric_qmin = -((qmax - qmin) / 2 + 1);
      int symmetric_qmax = (qmax - qmin) / 2;

      float max_scale = std::max(
          std::fabs(min_val / symmetric_qmin),
          std::fabs(max_val / symmetric_qmax));
      min_val = max_scale * symmetric_qmin;
      max_val = max_scale * symmetric_qmax;
    }

    // We extend the [min, max] interval to ensure that it contains 0.
    // Otherwise, we would not meet the requirement that 0 be an exactly
    // representable value.
    min_val = std::min(min_val, 0.f);
    max_val = std::max(max_val, 0.f);
    scale[i] = (static_cast<double>(max_val) - min_val) / (qmax - qmin);

    // Moving this check outside this function would result in extra Device to
    // Host copy of the min and max val which would result in a perf hit.
    if (scale[i] == 0.0f || std::isinf(1.0f / scale[i])) {
      scale[i] = 0.1;
    }

    double zero_point_from_min = qmin - min_val / static_cast<double>(scale[i]);
    double zero_point_from_max = qmax - max_val / static_cast<double>(scale[i]);
    double zero_point_from_min_error =
        std::abs(qmin) + std::abs(min_val / static_cast<double>(scale[i]));
    double zero_point_from_max_error =
        std::abs(qmax) + std::abs(max_val / static_cast<double>(scale[i]));
    double initial_zero_point =
        zero_point_from_min_error < zero_point_from_max_error
        ? zero_point_from_min
        : zero_point_from_max;

    // Note: preserve_sparsity here means symmetric quantization.
    // for symmetric quantization, we force zero_point
    // to be a middle value between qmin and qmax.
    // If either min or max is 0, then we just use 0 as zero_point.
    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      initial_zero_point = static_cast<double>(qmin + qmax) / 2;
    }
    // Now we need to nudge the zero point to be an integer
    // (our zero points are integer, and this is motivated by the
    // requirement to be able to represent the real value "0" exactly as a
    // quantized value, which is required in multiple places, for example in
    // Im2col with zero padding).
    int32_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
      nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
      nudged_zero_point = qmax;
    } else {
      nudged_zero_point = std::nearbyint(initial_zero_point);
    }
    zero_point[i] = nudged_zero_point;
  }
}

template <typename scalar_t>
struct CalcMovingAvgQparamsHelperKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    ChooseQuantizationParamsKernelImpl<scalar_t>(
        fake_quant_on_data_,
        running_min_data_,
        running_max_data_,
        qmin_,
        qmax_,
        size_,
        symmetric_quant_, // preserve_sparsity
        scale_ptr_,
        zp_ptr_,
        item);
  }
  CalcMovingAvgQparamsHelperKernelFunctor(
      const int64_t* fake_quant_on_data,
      const scalar_t* running_min_data,
      const scalar_t* running_max_data,
      int32_t qmin,
      int32_t qmax,
      int size,
      bool symmetric_quant,
      float* scale_ptr,
      int32_t* zp_ptr)
      : fake_quant_on_data_(fake_quant_on_data),
        running_min_data_(running_min_data),
        running_max_data_(running_max_data),
        qmin_(qmin),
        qmax_(qmax),
        size_(size),
        symmetric_quant_(symmetric_quant),
        scale_ptr_(scale_ptr),
        zp_ptr_(zp_ptr) {}

 private:
  const int64_t* fake_quant_on_data_;
  const scalar_t* running_min_data_;
  const scalar_t* running_max_data_;
  int32_t qmin_;
  int32_t qmax_;
  int size_;
  bool symmetric_quant_;
  float* scale_ptr_;
  int32_t* zp_ptr_;
};

void _calc_moving_avg_qparams_helper(
    const at::Tensor& x,
    const at::Tensor fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    float* scale_ptr,
    int32_t* zp_ptr,
    int32_t qmin,
    int32_t qmax,
    bool symmetric_quant,
    const int64_t size,
    bool per_row_fq = false) {
  auto execution_policy = calc_execution_policy(size);
  // auto counter_offset = std::get<0>(execution_policy);
  auto num_groups = std::get<1>(execution_policy);
  auto group_size = std::get<2>(execution_policy);

  int64_t* fake_quant_on_data = fake_quant_on.data_ptr<int64_t>();
  auto local_range = per_row_fq ? group_size : 1;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16,
      at::kHalf,
      x.scalar_type(),
      "ChooseQuantizationParams",
      [&] {
        scalar_t* running_min_data = running_min.data_ptr<scalar_t>();
        scalar_t* running_max_data = running_max.data_ptr<scalar_t>();

        CalcMovingAvgQparamsHelperKernelFunctor<scalar_t> kfn(
            fake_quant_on_data,
            running_min_data,
            running_max_data,
            qmin,
            qmax,
            size,
            symmetric_quant,
            scale_ptr,
            zp_ptr);
        sycl_kernel_submit(
            num_groups * group_size, local_range, getCurrentSYCLQueue(), kfn);
      });
}

} // namespace at::native::xpu
