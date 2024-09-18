#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/xpu/XPUNativeFunctions.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/empty_like.h>
#include <ATen/ops/fake_quantize_per_tensor_affine.h>
#endif

#include <ATen/native/quantized/xpu/sycl/FakeQuantizeCoreKernels.h>

namespace at {

std::tuple<Tensor, Tensor> XPUNativeFunctions::
    _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        const Tensor& self,
        const Tensor& scale,
        const Tensor& zero_point,
        const Tensor& fake_quant_enabled,
        int64_t quant_min,
        int64_t quant_max) {
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);
  native::xpu::_fake_quantize_tensor_cachemask_tensor_qparams_kernel(
      Y,
      mask,
      self,
      scale,
      zero_point,
      fake_quant_enabled,
      quant_min,
      quant_max);
  // TODO(future, optional): look into packing the mask further (BoolTensor uses
  //   1 byte per element, we only need 1 bit per element).
  return std::make_tuple(Y, mask);
}

static int64_t _get_zero_point_from_tensor(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    bool is_forward) {
  float zero_point_fp = zero_point[0].item<float>();
  zero_point_fp =
      is_forward ? std::nearbyint(zero_point_fp) : zero_point_fp + 0.5f;
  float zero_point_clamped = std::min(
      std::max(zero_point_fp, static_cast<float>(quant_min)),
      static_cast<float>(quant_max));
  return static_cast<int64_t>(zero_point_clamped);
}

Tensor XPUNativeFunctions::_fake_quantize_learnable_per_tensor_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  float scale_val = scale[0].item<float>();
  int64_t zero_point_val =
      _get_zero_point_from_tensor(zero_point, quant_min, quant_max, true);
  return fake_quantize_per_tensor_affine(
      self, scale_val, zero_point_val, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor> XPUNativeFunctions::
    _fake_quantize_learnable_per_tensor_affine_backward(
        const Tensor& dY,
        const Tensor& X,
        const Tensor& scale,
        const Tensor& zero_point,
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
  float scale_val = scale[0].item<float>();
  float inv_scale_val = 1.0f / scale_val;
  int64_t zero_point_val =
      _get_zero_point_from_tensor(zero_point, quant_min, quant_max, false);

  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float);
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.numel() == dY.numel(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= 0 && quant_max >= 0,
      "`quant_min` should be less than or \
        equal to `quant_max`, and the quantization range should include 0.");
  TORCH_CHECK(
      zero_point_val >= quant_min && zero_point_val <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");
  if (X.numel() <= 0) {
    return std::make_tuple(X, scale, zero_point);
  }

  auto dX = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dScale_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dZeroPoint_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);

  auto iter = TensorIteratorConfig()
                  .add_output(dX)
                  .add_output(dScale_vec)
                  .add_output(dZeroPoint_vec)
                  .add_input(X)
                  .add_input(dY)
                  .build();

  native::xpu::_fake_quantize_grad_learnable_tensor_kernel(
      iter,
      scale_val,
      inv_scale_val,
      zero_point_val,
      quant_min,
      quant_max,
      grad_factor);

  // The total sums over the scale and zero point gradient vectors are what will
  // be returned in the end.
  auto dScale = dScale_vec.sum().unsqueeze(0).to(scale.device());
  auto dZeroPoint = dZeroPoint_vec.sum().unsqueeze(0).to(zero_point.device());

  return std::make_tuple(dX, dScale, dZeroPoint);
}

} // namespace at