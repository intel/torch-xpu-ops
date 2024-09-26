#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/quantized/Quantizer.h>

#include <ATen/native/quantized/xpu/sycl/AffineQuantizerKernels.h>

namespace at::native {

static void checkRoundingMode(const std::string& fn_name) {
  // Disabling this warning message for now as it is printed incorrectly. Need
  // to fix

  /*  TORCH_WARN_ONCE(
        std::fegetround() != FE_TONEAREST,
        fn_name,
        " current rounding mode is not set to round-to-nearest-ties-to-even
     (FE_TONEAREST). This will cause accuracy issues in quantized models.");
  */
  return;
}

static void checkFloatTensor(const std::string& fn_name, const Tensor& t) {
  TORCH_CHECK(
      t.scalar_type() == kFloat,
      fn_name,
      " expects a Float Tensor, got ",
      t.scalar_type());
}

static void checkSameDevice(
    const std::string& fn_name,
    const Tensor& t1,
    const Tensor& t2) {
  TORCH_CHECK(
      t1.device() == t2.device(),
      fn_name,
      " expects a quantized and float tensors to be on the same device.");
}

template <typename T>
static void checkQuantizedTensor(const std::string& fn_name, const Tensor& t) {
  TORCH_CHECK(t.is_quantized(), fn_name, " expects a quantized Tensor.");
  TORCH_CHECK(
      t.scalar_type() == caffe2::TypeMeta::Make<T>(),
      fn_name,
      " expects a ",
      caffe2::TypeMeta::Make<T>(),
      " Tensor, got ",
      t.scalar_type());
}

template <typename T>
static void checkZeroPoint(const std::string& fn_name, int64_t zero_point) {
  TORCH_CHECK(
      zero_point <= std::numeric_limits<T>::max(),
      fn_name,
      " zero_point ",
      zero_point,
      " is above upper bound.");
  TORCH_CHECK(
      zero_point >= std::numeric_limits<T>::min(),
      fn_name,
      " zero_point ",
      zero_point,
      " is below lower bound.");
}

static void checkSameSize(
    const std::string& fn_name,
    const Tensor& qt,
    const Tensor& rt) {
  TORCH_CHECK(
      qt.sizes().equals(rt.sizes()),
      fn_name,
      " only works with Tensors with the same shape");
}

static void checkPerChannelParamsSize(
    const Tensor& rtensor,
    int64_t axis,
    const Tensor& scales,
    const Tensor& zero_points) {
  int64_t channel = rtensor.size(axis);
  TORCH_CHECK(
      channel == int64_t(scales.numel()),
      "length of scales must equal to channel, expected ",
      channel,
      " got, ",
      scales.numel());
  TORCH_CHECK(
      channel == int64_t(zero_points.numel()),
      "length of zero_points must equal to channel expected ",
      channel,
      " got, ",
      zero_points.numel());
}

Tensor& quantize_tensor_per_channel_affine_xpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "quantize_tensor_per_channel_affine_xpu";

  checkRoundingMode(fn_name);
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
  });

  TORCH_CHECK(
      0 <= axis && axis < rtensor.dim(),
      "Channel axis out of range in per channel affine quantization. Got: ",
      axis,
      "Expected: [0, ",
      rtensor.dim(),
      ")");
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  native::xpu::quantize_tensor_per_channel_affine_kernel(
      rtensor, qtensor, scales, zero_points, axis);
  return qtensor;
}

Tensor& dequantize_tensor_per_channel_affine_xpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_affine_xpu";

  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
  });

  TORCH_CHECK(
      0 <= axis && axis < qtensor.dim(),
      "Channel axis out of range in per channel affine dequantization. Got:",
      axis,
      " Expected: [0, ",
      qtensor.dim(),
      ")");
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  native::xpu::dequantize_tensor_per_channel_affine_kernel(
      qtensor, rtensor, scales, zero_points, axis);
  return rtensor;
}

Tensor& quantize_tensor_per_channel_float_qparams_xpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name =
      "quantize_tensor_per_channel_float_qparams_xpu";

  checkRoundingMode(fn_name);
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
  });

  TORCH_CHECK(
      0 <= axis && axis < rtensor.dim(),
      "Channel axis out of range in per channel float qparams quantization. Got: ",
      axis,
      "Expected: [0, ",
      rtensor.dim(),
      ")");
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  native::xpu::quantize_tensor_per_channel_float_qparams_kernel(
      rtensor, qtensor, scales, zero_points, axis);
  return qtensor;
}

Tensor& dequantize_tensor_per_channel_float_qparams_xpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_affine_xpu";

  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
  });

  TORCH_CHECK(
      0 <= axis && axis < qtensor.dim(),
      "Channel axis out of range in per channel float qparams dequantization. Got:",
      axis,
      " Expected: [0, ",
      qtensor.dim(),
      ")");
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  native::xpu::dequantize_tensor_per_channel_float_qparams_kernel(
      qtensor, rtensor, scales, zero_points, axis);
  return rtensor;
}

Tensor& quantize_tensor_per_tensor_affine_xpu(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  static constexpr auto fn_name = "quantize_tensor_per_tensor_affine_xpu";

  checkRoundingMode(fn_name);
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    checkZeroPoint<underlying_t>(fn_name, zero_point);
  });

  native::xpu::quantize_tensor_per_tensor_affine_kernel(
      rtensor, qtensor, scale, zero_point);
  return qtensor;
}

Tensor& dequantize_tensor_per_tensor_affine_xpu(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  static constexpr auto fn_name = "dequantize_tensor_per_tensor_affine_xpu";
  checkFloatTensor(fn_name, rtensor);
  checkSameDevice(fn_name, rtensor, qtensor);
  checkSameSize(fn_name, qtensor, rtensor);

  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    checkZeroPoint<underlying_t>(fn_name, zero_point);
  });

  native::xpu::dequantize_tensor_per_tensor_affine_kernel(
      qtensor, rtensor, scale, zero_point);
  return rtensor;
}

} // namespace at::native
