#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/quantized/Quantizer.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <ATen/native/quantized/xpu/AffineQuantizer.h>
#include <ATen/native/quantized/xpu/Quantizer.h>

namespace at::native {

struct PerChannelAffineQuantizerXPU : public AffineQuantizer {
  explicit PerChannelAffineQuantizerXPU(
      ScalarType scalar_type,
      Tensor scales,
      Tensor zero_points,
      int64_t axis)
      : AffineQuantizer(scalar_type),
        scales_(scales),
        zero_points_(zero_points),
        axis_(axis) {}

  QScheme qscheme() const override {
    return kPerChannelAffine;
  }

  Tensor scales() const {
    return scales_;
  }

  Tensor zero_points() const {
    return zero_points_;
  }

  int64_t axis() const {
    return axis_;
  }

  Tensor quantize(const Tensor& rtensor) override {
    TORCH_CHECK(
        rtensor.scalar_type() == kFloat,
        "quantize only works on Float Tensor.");

    Tensor qtensor = native::xpu::new_qtensor(
        rtensor.sizes(),
        rtensor.options()
            .dtype(scalar_type_)
            .memory_format(rtensor.suggest_memory_format()),
        intrusive_from_this());

    auto rtensor_contig = rtensor.contiguous(rtensor.suggest_memory_format());

    return quantize_tensor_per_channel_affine_xpu(
        qtensor, rtensor_contig, scales_, zero_points_, axis_);
  }

  Tensor dequantize(const Tensor& qtensor) override {
    if (!qtensor.is_quantized()) {
      return qtensor;
    }

    Tensor rtensor = at::empty(
        qtensor.sizes(),
        qtensor.options()
            .dtype(at::kFloat)
            .memory_format(qtensor.suggest_memory_format()));
    auto qtensor_contig = qtensor.contiguous(qtensor.suggest_memory_format());

    return dequantize_tensor_per_channel_affine_xpu(
        rtensor, qtensor_contig, scales_, zero_points_, axis_);
  }

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerChannelAffine) {
      return false;
    }
    auto* other_per_channel_affine =
        static_cast<PerChannelAffineQuantizerXPU*>(other.get());
    return scalar_type() == other_per_channel_affine->scalar_type() &&
        scales().equal(other_per_channel_affine->scales()) &&
        zero_points().equal(other_per_channel_affine->zero_points()) &&
        axis() == other_per_channel_affine->axis();
  }

  Tensor& dequantize_out(Tensor& out, const Tensor& t) override {
    TORCH_CHECK(false, "not implemented");
  }

 private:
  Tensor scales_;
  Tensor zero_points_;
  const int64_t axis_;
};

QuantizerPtr make_per_channel_affine_quantizer_xpu(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type) {
  TORCH_CHECK(
      scales.numel() == zero_points.numel(),
      "number of elements in scales and zero_points must match");
  TORCH_CHECK(
      isFloatingType(scales.scalar_type()),
      "scale tensor must be floating point");
  TORCH_CHECK(
      isIntegralType(zero_points.scalar_type(), false /*includeBool*/),
      "zero_points tensor must have integral type");
  Tensor scales_double = scales.to(kFloat).contiguous();
  Tensor zero_points_int64 = zero_points.to(kInt).contiguous();
  return c10::make_intrusive<PerChannelAffineQuantizerXPU>(
      scalar_type, scales_double, zero_points_int64, axis);
}

struct PerTensorAffineQuantizerXPU : public AffineQuantizer {
  explicit PerTensorAffineQuantizerXPU(
      ScalarType scalar_type,
      double scale,
      int64_t zero_point)
      : AffineQuantizer(scalar_type), scale_(scale), zero_point_(zero_point) {}

  Tensor quantize(const Tensor& rtensor) override {
    TORCH_CHECK(
        rtensor.scalar_type() == kFloat,
        "quantize only works on Float Tensor.");
    Tensor qtensor = native::xpu::new_qtensor(
        rtensor.sizes(),
        rtensor.options()
            .dtype(scalar_type_)
            .memory_format(rtensor.suggest_memory_format()),
        intrusive_from_this());
    auto rtensor_contig = rtensor.contiguous(rtensor.suggest_memory_format());

    return native::quantize_tensor_per_tensor_affine_xpu(
        qtensor, rtensor_contig, scale_, zero_point_);
  }

  Tensor dequantize(const Tensor& qtensor) override {
    if (!qtensor.is_quantized()) {
      return qtensor;
    }

    Tensor rtensor = at::empty(
        qtensor.sizes(),
        qtensor.options()
            .dtype(at::kFloat)
            .memory_format(qtensor.suggest_memory_format()));
    auto qtensor_contig = qtensor.contiguous(qtensor.suggest_memory_format());

    return native::dequantize_tensor_per_tensor_affine_xpu(
        rtensor, qtensor_contig, scale_, zero_point_);
  }

  QScheme qscheme() const override {
    return kPerTensorAffine;
  }

  double scale() const {
    return scale_;
  }

  int64_t zero_point() const {
    return zero_point_;
  }

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerTensorAffine) {
      return false;
    }
    auto* other_per_tensor_affine =
        static_cast<PerTensorAffineQuantizerXPU*>(other.get());
    return scalar_type() == other_per_tensor_affine->scalar_type() &&
        scale() == other_per_tensor_affine->scale() &&
        zero_point() == other_per_tensor_affine->zero_point();
  }

  Tensor& dequantize_out(Tensor& out, const Tensor& t) override {
    TORCH_CHECK(false, "not implemented");
  }

 private:
  const double scale_;
  // We use int64_t for consistency with Python
  const int64_t zero_point_;
};

QuantizerPtr make_per_tensor_affine_quantizer_xpu(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type) {
  return c10::make_intrusive<PerTensorAffineQuantizerXPU>(
      scalar_type, scale, zero_point);
}

} // namespace at::native
