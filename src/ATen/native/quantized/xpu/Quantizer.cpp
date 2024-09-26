#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/quantized/QTensorImpl.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/Allocator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#endif

#include <ATen/native/quantized/xpu/AffineQuantizer.h>

namespace at::native::xpu {

static int64_t get_sub_byte_tensor_size(
    IntArrayRef sizes,
    size_t dtype_itemsize,
    at::ScalarType t) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t element_per_byte;
  switch (t) {
    case at::ScalarType::QUInt4x2:
      element_per_byte = 2;
      break;
    case at::ScalarType::QUInt2x4:
      element_per_byte = 4;
      break;
    default:
      element_per_byte = 1;
  }
  // zero dim tensor
  if (sizes.empty()) {
    return c10::multiply_integers(sizes) * dtype_itemsize;
  }
  // Consider most inner dim as cols
  int64_t cols = sizes.at(sizes.size() - 1);
  int64_t bytes_per_row = cols * dtype_itemsize;
  // align qtensor most inner dim, compute ceil (bytes_per_row /
  // element_per_byte)
  return c10::multiply_integers(IntArrayRef(sizes.data(), sizes.size() - 1)) *
      at::ceil_div(bytes_per_row, element_per_byte);
}

Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer) {
  at::DispatchKey tensorDispatchKey = options.computeDispatchKey();
  native::check_size_nonnegative(sizes);
  auto dtype = options.dtype();
  TORCH_CHECK(
      isQIntType(typeMetaToScalarType(dtype)),
      dtype,
      " is not supported in new_qtensor on xpu device.");
  auto scalar_type = typeMetaToScalarType(dtype);
  int64_t size_bytes =
      get_sub_byte_tensor_size(sizes, dtype.itemsize(), scalar_type);

  auto memory_format =
      options.memory_format_opt().value_or(MemoryFormat::Contiguous);
  at::Allocator* allocator = c10::GetAllocator(kXPU);

  auto storage = c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator->allocate(size_bytes),
      allocator,
      /*resizable=*/true);
  auto tensor = detail::make_tensor<QTensorImpl>(
      storage, at::DispatchKeySet(tensorDispatchKey), dtype, quantizer);

  get_qtensorimpl(tensor)->set_sizes_contiguous(sizes);
  get_qtensorimpl(tensor)->empty_tensor_restride(memory_format);

  return tensor;
}

static void per_channel_affine_dequantize_impl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scale,
    const Tensor& zero_point,
    const int64_t axis) {
  const auto qtensor_contig =
      qtensor.expect_contiguous(qtensor.suggest_memory_format());
  native::dequantize_tensor_per_channel_affine_xpu(
      *qtensor_contig, rtensor, scale, zero_point, axis);
}

struct XPUPerChannelAffineQuantizer : public AffineQuantizer {
  explicit XPUPerChannelAffineQuantizer(
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
    // Here we need a std::intrusive_ptr<Quantizer>.. but actually "this" is the
    // quantizer that can be reused, so I'm using intrusive_from_this here
    Tensor qtensor = native::xpu::new_qtensor(
        rtensor.sizes(),
        rtensor.options()
            .dtype(scalar_type_)
            .memory_format(rtensor.suggest_memory_format()),
        intrusive_from_this());
    auto rtensor_contig =
        rtensor.expect_contiguous(rtensor.suggest_memory_format());
    native::quantize_tensor_per_channel_affine_xpu(
        *rtensor_contig, qtensor, scales_, zero_points_, axis_);
    return qtensor;
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
    per_channel_affine_dequantize_impl(
        rtensor, qtensor, scales_, zero_points_, axis_);
    return rtensor;
  }

  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override {
    rtensor.resize_(qtensor.sizes());
    TORCH_CHECK(
        rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
            rtensor.scalar_type() == kFloat,
        "Dequantize out should be a contiguous Float Tensor; instead got type ",
        rtensor.scalar_type(),
        ", and is_contiguous ",
        rtensor.is_contiguous(qtensor.suggest_memory_format()));
    per_channel_affine_dequantize_impl(
        rtensor, qtensor, scales_, zero_points_, axis_);
    return rtensor;
  }

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerChannelAffine) {
      return false;
    }
    auto* other_per_channel_affine =
        static_cast<XPUPerChannelAffineQuantizer*>(other.get());
    return scalar_type() == other_per_channel_affine->scalar_type() &&
        scales().equal(other_per_channel_affine->scales()) &&
        zero_points().equal(other_per_channel_affine->zero_points()) &&
        axis() == other_per_channel_affine->axis();
  }

 protected:
  Tensor scales_;
  Tensor zero_points_;
  const int64_t axis_;
};

static void per_channel_affine_float_q_params_dequantize_impl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    const int64_t axis) {
  const auto qtensor_contig =
      qtensor.expect_contiguous(qtensor.suggest_memory_format());
  dequantize_tensor_per_channel_float_qparams_xpu(
      *qtensor_contig, rtensor, scales, zero_points, axis);
}

struct XPUPerChannelAffineFloatQParamsQuantizer
    : public XPUPerChannelAffineQuantizer {
  explicit XPUPerChannelAffineFloatQParamsQuantizer(
      ScalarType scalar_type,
      Tensor scales,
      Tensor zero_points,
      int64_t axis)
      : XPUPerChannelAffineQuantizer(scalar_type, scales, zero_points, axis) {}

  QScheme qscheme() const override {
    return kPerChannelAffineFloatQParams;
  }

  Tensor quantize(const Tensor& rtensor) override {
    TORCH_CHECK(
        rtensor.scalar_type() == kFloat,
        "Quantize only works on Float Tensor, got ",
        rtensor.scalar_type());
    Tensor qtensor = native::xpu::new_qtensor(
        rtensor.sizes(),
        rtensor.options().dtype(scalar_type_),
        intrusive_from_this());
    auto rtensor_contig = rtensor.expect_contiguous();
    quantize_tensor_per_channel_float_qparams_xpu(
        *rtensor_contig, qtensor, scales_, zero_points_, axis_);
    return qtensor;
  }

  Tensor dequantize(const Tensor& qtensor) override {
    if (!qtensor.is_quantized()) {
      return qtensor;
    }
    Tensor rtensor =
        at::empty(qtensor.sizes(), qtensor.options().dtype(at::kFloat));
    per_channel_affine_float_q_params_dequantize_impl(
        rtensor, qtensor, scales_, zero_points_, axis_);
    return rtensor;
  }

  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override {
    rtensor.resize_(qtensor.sizes());
    TORCH_CHECK(
        rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
            rtensor.scalar_type() == kFloat,
        "Dequantize out should be a contiguous Float Tensor; instead got type ",
        rtensor.scalar_type(),
        ", and is_contiguous ",
        rtensor.is_contiguous(qtensor.suggest_memory_format()));
    per_channel_affine_float_q_params_dequantize_impl(
        rtensor, qtensor, scales_, zero_points_, axis_);
    return rtensor;
  }

  bool equalTo(QuantizerPtr other) const override {
    if (!other.get() || other->qscheme() != kPerChannelAffineFloatQParams) {
      return false;
    }
    auto* other_per_channel_float_qparams =
        static_cast<XPUPerChannelAffineFloatQParamsQuantizer*>(other.get());
    return scalar_type() == other_per_channel_float_qparams->scalar_type() &&
        scales().equal(other_per_channel_float_qparams->scales()) &&
        zero_points().equal(other_per_channel_float_qparams->zero_points()) &&
        axis() == other_per_channel_float_qparams->axis();
  }
};

static void checkPerChannelParamDims(
    const Tensor& scales,
    const Tensor& zero_points) {
  TORCH_CHECK(scales.dim() == 1, "scale tensor must have dimension 1");
  TORCH_CHECK(
      zero_points.dim() == 1, "zero_points tensor must have dimension 1");
  TORCH_CHECK(
      scales.numel() == zero_points.numel(),
      "number of elements in scales and zero_points must match");
}

QuantizerPtr make_per_channel_affine_quantizer(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type) {
  checkPerChannelParamDims(scales, zero_points);
  TORCH_CHECK(
      isFloatingType(scales.scalar_type()),
      "scale tensor must be floating point");
  if (isFloatingType(zero_points.scalar_type())) {
    Tensor scales_float = scales.to(kFloat).contiguous();
    Tensor zero_points_float = zero_points.to(kFloat).contiguous();
    return c10::make_intrusive<XPUPerChannelAffineFloatQParamsQuantizer>(
        scalar_type, scales_float, zero_points_float, axis);
  } else {
    Tensor scales_double = scales.to(kDouble).contiguous();
    Tensor zero_points_int64 = zero_points.to(kLong).contiguous();
    return c10::make_intrusive<XPUPerChannelAffineQuantizer>(
        scalar_type, scales_double, zero_points_int64, axis);
  }
}

static void per_tensor_affine_dequantize_impl(
    Tensor& rtensor,
    const Tensor& qtensor,
    const double scale,
    const int64_t zero_point) {
  const auto qtensor_contig =
      qtensor.expect_contiguous(qtensor.suggest_memory_format());
  native::dequantize_tensor_per_tensor_affine_xpu(
      *qtensor_contig, rtensor, scale, zero_point);
}

struct XPUPerTensorAffineQuantizer : public AffineQuantizer {
  explicit XPUPerTensorAffineQuantizer(
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

    auto rtensor_contig =
        rtensor.expect_contiguous(rtensor.suggest_memory_format());
    native::quantize_tensor_per_tensor_affine_xpu(
        *rtensor_contig, qtensor, scale_, zero_point_);
    return qtensor;
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
    per_tensor_affine_dequantize_impl(rtensor, qtensor, scale_, zero_point_);
    return rtensor;
  }

  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override {
    rtensor.resize_(qtensor.sizes());
    TORCH_CHECK(
        rtensor.is_contiguous(qtensor.suggest_memory_format()) &&
            rtensor.scalar_type() == kFloat,
        "Dequantize out should be a contiguous Float Tensor; instead got type ",
        rtensor.scalar_type(),
        ", and is_contiguous ",
        rtensor.is_contiguous(qtensor.suggest_memory_format()));
    per_tensor_affine_dequantize_impl(rtensor, qtensor, scale_, zero_point_);
    return rtensor;
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
        static_cast<XPUPerTensorAffineQuantizer*>(other.get());
    return scalar_type() == other_per_tensor_affine->scalar_type() &&
        scale() == other_per_tensor_affine->scale() &&
        zero_point() == other_per_tensor_affine->zero_point();
  }

 private:
  const double scale_;
  // We use int64_t for consistency with Python
  const int64_t zero_point_;
};

QuantizerPtr make_per_tensor_affine_quantizer(
    double scale,
    int64_t zero_point,
    ScalarType scalar_type) {
  return c10::make_intrusive<XPUPerTensorAffineQuantizer>(
      scalar_type, scale, zero_point);
}

} // namespace at::native::xpu
