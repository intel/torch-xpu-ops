#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/any.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/lt.h>
#endif

#include <ATen/native/quantized/sycl/AffineQuantizerKernels.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename T>
static void check_zero_points_xpu(
    const std::string& fn_name,
    const Tensor& zero_points) {
  constexpr int64_t qmin = std::numeric_limits<T>::min();
  constexpr int64_t qmax = std::numeric_limits<T>::max();
  auto zp_within_upper = at::any(at::gt(zero_points, qmax)).item().equal(false);
  auto zp_within_lower = at::any(at::lt(zero_points, qmin)).item().equal(false);
  TORCH_CHECK(zp_within_lower, fn_name, "zero_point is below lower bound.");
  TORCH_CHECK(zp_within_upper, fn_name, "zero_point is above upper bound.");
}

template <typename scalar_t>
struct QuantizerTensorPerChannelAffineFunctor {
  scalar_t operator()(
      float raw_val,
      scalar_t quantized_val,
      double scale,
      int64_t zero_point) const {
    int64_t qvalue =
        static_cast<int64_t>(std::nearbyint(raw_val / scale) + zero_point);
    qvalue = std::max<int64_t>(qvalue, qmin_);
    qvalue = std::min<int64_t>(qvalue, qmax_);
    quantized_val.val_ = qvalue;
    return quantized_val;
  }

  QuantizerTensorPerChannelAffineFunctor(int64_t qmin, int64_t qmax)
      : qmin_(qmin), qmax_(qmax) {}

 private:
  const int64_t qmin_;
  const int64_t qmax_;
};

void quantize_tensor_per_channel_affine_kernel(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "quantize_tensor_per_channel_affine_xpu";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(qtensor)
                  .add_input(rtensor)
                  .add_input(qtensor)
                  .add_input(shaped_scales)
                  .add_input(shaped_zero_points)
                  .build();

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    check_zero_points_xpu<underlying_t>(fn_name, zero_points);

    constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
    constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();
    // trying to match _quantize_per_channel_ref_nd in test_quantized_tensor.py
    auto caller = QuantizerTensorPerChannelAffineFunctor<scalar_t>(qmin, qmax);
    gpu_kernel(iter, caller);
  });
}

template <typename scalar_t>
struct DequantizerTensorPerChannelAffineFunctor {
  float operator()(scalar_t value, double scale, int64_t zero_point) const {
    return static_cast<float>(value.val_ - zero_point) * scale;
  }

  DequantizerTensorPerChannelAffineFunctor() {}
};

void dequantize_tensor_per_channel_affine_kernel(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_affine_xpu";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    check_zero_points_xpu<underlying_t>(fn_name, zero_points);

    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(false)
                    .add_output(rtensor)
                    .add_input(qtensor)
                    .add_input(shaped_scales)
                    .add_input(shaped_zero_points)
                    .build();

    gpu_kernel(iter, DequantizerTensorPerChannelAffineFunctor<scalar_t>());
  });
}

template <typename scalar_t>
struct QuantizerTensorPerChannelFloatQparamsFunctor {
  scalar_t operator()(
      float raw_val,
      scalar_t quantized_val,
      float scale,
      float zero_point) const {
    float inv_scale = 1.0f / scale;
    int64_t qvalue =
        static_cast<int64_t>(rintf(raw_val * inv_scale + zero_point));
    qvalue = std::max<int64_t>(qvalue, qmin_);
    qvalue = std::min<int64_t>(qvalue, qmax_);
    quantized_val.val_ = qvalue;
    return quantized_val;
  }

  QuantizerTensorPerChannelFloatQparamsFunctor(int64_t qmin, int64_t qmax)
      : qmin_(qmin), qmax_(qmax) {}

 private:
  const int64_t qmin_;
  const int64_t qmax_;
};

void quantize_tensor_per_channel_float_qparams_kernel(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name =
      "quantize_tensor_per_channel_float_qparams_xpu";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(qtensor)
                  .add_input(rtensor)
                  .add_input(qtensor)
                  .add_input(shaped_scales)
                  .add_input(shaped_zero_points)
                  .build();

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    check_zero_points_xpu<underlying_t>(fn_name, zero_points);

    constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
    constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();

    auto caller =
        QuantizerTensorPerChannelFloatQparamsFunctor<scalar_t>(qmin, qmax);
    // trying to match _quantize_per_channel_ref_nd in
    gpu_kernel(iter, caller);
  });
}

template <typename scalar_t>
struct DequantizerTensorPerChannelFloatQparamsFunctor {
  float operator()(scalar_t value, float scale, float zero_point) const {
    return static_cast<float>(static_cast<float>(value.val_) - zero_point) *
        scale;
  }

  DequantizerTensorPerChannelFloatQparamsFunctor() {}
};

void dequantize_tensor_per_channel_float_qparams_kernel(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name =
      "dequantize_tensor_per_channel_float_qparams_xpu";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    check_zero_points_xpu<underlying_t>(fn_name, zero_points);

    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(false)
                    .add_output(rtensor)
                    .add_input(qtensor)
                    .add_input(shaped_scales)
                    .add_input(shaped_zero_points)
                    .build();

    gpu_kernel(
        iter, DequantizerTensorPerChannelFloatQparamsFunctor<scalar_t>());
  });
}

template <typename scalar_t>
struct QuantizerTensorPerTensorAffineFunctor {
  scalar_t operator()(float raw_val, scalar_t quantized_val) const {
    int64_t qvalue =
        static_cast<int64_t>(std::nearbyint(raw_val / scale_) + zero_point_);
    qvalue = std::max<int64_t>(qvalue, qmin_);
    qvalue = std::min<int64_t>(qvalue, qmax_);
    quantized_val.val_ = qvalue;
    return quantized_val;
  }

  QuantizerTensorPerTensorAffineFunctor(
      int64_t qmin,
      int64_t qmax,
      double scale,
      int64_t zero_point)
      : qmin_(qmin), qmax_(qmax), scale_(scale), zero_point_(zero_point) {}

 private:
  const int64_t qmin_;
  const int64_t qmax_;
  const double scale_;
  const int64_t zero_point_;
};

void quantize_tensor_per_tensor_affine_kernel(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_xpu", [&]() {
        constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
        constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();

        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(qtensor)
                        .add_input(rtensor)
                        .add_input(qtensor)
                        .build();
        auto caller = QuantizerTensorPerTensorAffineFunctor<scalar_t>(
            qmin, qmax, scale, zero_point);
        gpu_kernel(iter, caller);
      });
}

template <typename scalar_t>
struct DequantizerTensorPerTensorAffineFunctor {
  float operator()(scalar_t value) const {
    return static_cast<float>(static_cast<float>(value.val_) - zero_point_) *
        scale_;
  }

  DequantizerTensorPerTensorAffineFunctor(double scale, int64_t zero_point)
      : scale_(scale), zero_point_(zero_point) {}

 private:
  const double scale_;
  const int64_t zero_point_;
};

void dequantize_tensor_per_tensor_affine_kernel(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_xpu", [&]() {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(rtensor)
                        .add_input(qtensor)
                        .build();

        auto caller = DequantizerTensorPerTensorAffineFunctor<scalar_t>(
            scale, zero_point);
        gpu_kernel(iter, caller);
      });
}

} // namespace at::native::xpu
