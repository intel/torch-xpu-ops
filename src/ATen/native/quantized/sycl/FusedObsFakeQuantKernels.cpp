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

void MovingAverageMinMax(
    const int64_t* observer_on,
    const float* x_min,
    const float* x_max,
    float* running_min,
    float* running_max,
    const float averaging_const,
    const int size,
    sycl::nd_item<1>& item) {
  int i = item.get_group(0) * item.get_local_range(0) + item.get_local_id(0);

  if (*observer_on == 1) {
    if (i < size) {
      float curr_min = x_min[i];
      float curr_max = x_max[i];

      float adjusted_min = std::isinf(running_min[i])
          ? curr_min
          : (running_min[i]) + averaging_const * (curr_min - (running_min[i]));

      float adjusted_max = std::isinf(running_max[i])
          ? curr_max
          : (running_max[i]) + averaging_const * (curr_max - (running_max[i]));

      running_min[i] = adjusted_min;
      running_max[i] = adjusted_max;
    }
  }
}

struct CalculateMovingAverageKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    MovingAverageMinMax(
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
      const float* x_min_data,
      const float* x_max_data,
      float* running_min_data,
      float* running_max_data,
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
  const float* x_min_data_;
  const float* x_max_data_;
  float* running_min_data_;
  float* running_max_data_;
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
  float* running_min_data = running_min.data_ptr<float>();
  float* running_max_data = running_max.data_ptr<float>();

  if (per_row_fake_quant) {
    std::tie(x_min, x_max) = at::_aminmax(x, 1);
    float* x_min_data = x_min.data_ptr<float>();
    float* x_max_data = x_max.data_ptr<float>();

    // Moving Average Min/Max observer for activations
    CalculateMovingAverageKernelFunctor kfn(
        observer_on_data,
        x_min_data,
        x_max_data,
        running_min_data,
        running_max_data,
        averaging_const,
        size);
    sycl_kernel_submit(
        num_groups * group_size, group_size, getCurrentSYCLQueue(), kfn);
  } else {
    std::tie(x_min, x_max) = at::_aminmax(x);
    float* x_min_data = x_min.data_ptr<float>();
    float* x_max_data = x_max.data_ptr<float>();

    // Moving Average Min/Max observer for activations
    CalculateMovingAverageKernelFunctor kfn(
        observer_on_data,
        x_min_data,
        x_max_data,
        running_min_data,
        running_max_data,
        averaging_const,
        size);
    sycl_kernel_submit(num_groups * group_size, 1, getCurrentSYCLQueue(), kfn);
  }
}

void ChooseQuantizationParamsKernelImpl(
    const int64_t* fake_quant_on,
    const float* x_min,
    const float* x_max,
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
    scale[i] = (max_val - min_val) / (qmax - qmin);

    // Moving this check outside this function would result in extra Device to
    // Host copy of the min and max val which would result in a perf hit.
    if (scale[i] == 0.0f || std::isinf(1.0f / scale[i])) {
      scale[i] = 0.1;
    }

    float zero_point_from_min = qmin - min_val / scale[i];
    float zero_point_from_max = qmax - max_val / scale[i];
    float zero_point_from_min_error =
        std::abs(qmin) + std::abs(min_val / scale[i]);
    float zero_point_from_max_error =
        std::abs(qmax) + std::abs(max_val / scale[i]);
    float initial_zero_point =
        zero_point_from_min_error < zero_point_from_max_error
        ? zero_point_from_min
        : zero_point_from_max;

    // Note: preserve_sparsity here means symmetric quantization.
    // for symmetric quantization, we force zero_point
    // to be a middle value between qmin and qmax.
    // If either min or max is 0, then we just use 0 as zero_point.
    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      initial_zero_point = static_cast<float>(qmin + qmax) / 2;
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

struct CalcMovingAvgQparamsHelperKernelFunctor {
  void operator()(sycl::nd_item<1> item) const {
    ChooseQuantizationParamsKernelImpl(
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
      const float* running_min_data,
      const float* running_max_data,
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
  const float* running_min_data_;
  const float* running_max_data_;
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
  if (per_row_fq) {
    float* running_min_data = running_min.data_ptr<float>();
    float* running_max_data = running_max.data_ptr<float>();

    CalcMovingAvgQparamsHelperKernelFunctor kfn(
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
        num_groups * group_size, group_size, getCurrentSYCLQueue(), kfn);
  } else {
    float* running_min_data = running_min.data_ptr<float>();
    float* running_max_data = running_max.data_ptr<float>();
    CalcMovingAvgQparamsHelperKernelFunctor kfn(
        fake_quant_on_data,
        running_min_data,
        running_max_data,
        qmin,
        qmax,
        1, // size
        symmetric_quant, // preserve_sparsity
        scale_ptr,
        zp_ptr);
    sycl_kernel_submit(num_groups * group_size, 1, getCurrentSYCLQueue(), kfn);
  }
}

} // namespace at::native::xpu