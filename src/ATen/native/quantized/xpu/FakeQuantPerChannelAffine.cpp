#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/fake_quantize_per_channel_affine.h>
#endif

#include <ATen/native/quantized/xpu/sycl/FakeQuantizeCoreKernels.h>

namespace at {

static Tensor _get_rounded_zero_point(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // This assumes the per channel zero point vector is single-dimensioned.
  return zero_point.round().clamp_(quant_min, quant_max);
}

Tensor XPUNativeFunctions::_fake_quantize_learnable_per_channel_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  Tensor zero_point_rounded =
      _get_rounded_zero_point(zero_point, quant_min, quant_max).to(at::kInt);
  return fake_quantize_per_channel_affine(
      self, scale, zero_point_rounded, axis, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::
    _fake_quantize_learnable_per_channel_affine_backward(
        const Tensor& dY,
        const Tensor& X,
        const Tensor& scale,
        const Tensor& zero_point,
        int64_t axis,
        int64_t quant_min,
        int64_t quant_max,
        double grad_factor) {
  /* The gradients for scale and zero point are calculated as below:
     Let Xfq be the fake quantized version of X.
     Let Xq be the quantized version of X (clamped at qmin and qmax).
     Let Delta and z be the scale and the zero point.
     :math:
      \frac{d\Delta }{dx} =
        \begin{cases}
          q_{\min} - z& \text{ if } X_q= q_{\min} \\
          q_{\max} - z& \text{ if } X_q= q_{\max} \\
          (X_{fq} - X) / \Delta & \text{ else }
        \end{cases}

      \frac{dz }{dx} =
        \begin{cases}
          -\Delta& \text{ if } X_q= q_{\min} \text{ or } X_q = q_{\max} \\
          0 & \text{ else }
        \end{cases}
  */
  auto zero_point_rounded =
      _get_rounded_zero_point(zero_point, quant_min, quant_max);

  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float);
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Float);

  TORCH_CHECK(X.sizes() == dY.sizes(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= 0 && quant_max >= 0,
      "Expecting `quant_min` <= 0 and `quant_max` >= 0");
  TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor");
  TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor");
  TORCH_CHECK(
      scale.numel() == zero_point.numel(),
      "scale and zero-point need to have the same dimensions");
  TORCH_CHECK(
      scale.numel() == X.size(axis),
      "dimensions of scale and zero-point are not consistent with input tensor")

  TORCH_CHECK(
      at::min(zero_point_rounded).item().toLong() >= quant_min &&
          at::max(zero_point_rounded).item().toLong() <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  TORCH_CHECK(
      axis >= 0 && axis < X.dim(),
      "`axis` must be between 0 and number of dimensions of input");

  if (X.numel() <= 0) {
    return std::make_tuple(X, scale, zero_point);
  }

  auto dX = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dScale_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dZeroPoint_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto numDimensions = X.ndimension();

  // Create an axis mask for vectorizing and reshaping the scale and zero point
  // tensors into the same shapes as X along the channel axis.
  c10::DimVector axis_mask(numDimensions);
  for (const auto i : c10::irange(numDimensions)) {
    axis_mask[i] = (i == axis) ? X.size(axis) : 1;
  }
  auto X_shape = X.sizes();
  auto scale_vectorized =
      scale.reshape(at::IntArrayRef(axis_mask.data(), numDimensions))
          .expand(X_shape);
  auto zero_point_vectorized =
      zero_point_rounded
          .reshape(at::IntArrayRef(axis_mask.data(), numDimensions))
          .expand(X_shape);

  auto iter = TensorIteratorConfig()
                  .add_output(dX)
                  .add_output(dScale_vec)
                  .add_output(dZeroPoint_vec)
                  .add_input(X)
                  .add_input(dY)
                  .add_input(scale_vectorized)
                  .add_input(zero_point_vectorized)
                  .build();

  native::xpu::_fake_quantize_grad_learnable_channel_kernel(
      iter, quant_min, quant_max, grad_factor);

  auto numElements = X.ndimension() - 1;

  // Create a collection of axes that include all but the channel axis for
  // reduction when summing over the dScale and dZeroPoint tensors.
  c10::DimVector axis_for_reduction(numElements);
  for (const auto i : c10::irange(axis)) {
    axis_for_reduction[i] = i;
  }
  for (const auto i : c10::irange(axis, numElements)) {
    axis_for_reduction[i] = i + 1;
  }

  auto dScale =
      dScale_vec.sum(at::IntArrayRef(axis_for_reduction.data(), numElements));
  auto dZeroPoint = dZeroPoint_vec.sum(
      at::IntArrayRef(axis_for_reduction.data(), numElements));

  return std::make_tuple(dX, dScale, dZeroPoint);
}

} // namespace at