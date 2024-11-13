#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.h>
#include <ATen/ops/fake_quantize_per_channel_affine_cachemask.h>
#include <ATen/ops/ones_like.h>
#endif

#include <ATen/native/quantized/sycl/FusedObsFakeQuantKernels.h>
#include <ATen/native/xpu/sycl/NumericLimits.h>

namespace at {
namespace native {

std::tuple<at::Tensor, at::Tensor> fused_moving_avg_obs_fake_quant_xpu(
    const at::Tensor& x,
    const at::Tensor& observer_on,
    const at::Tensor& fake_quant_on,
    at::Tensor& running_min,
    at::Tensor& running_max,
    at::Tensor& scale,
    at::Tensor& zero_point,
    const double averaging_const,
    const int64_t quant_min,
    const int64_t quant_max,
    const int64_t ch_axis,
    bool per_row_fq,
    bool symmetric_quant) {
  TORCH_CHECK(
      ch_axis < x.dim(),
      "Error in fused_moving_avg_obs_fq_helper: ch_axis must be < "
      "self.dim()");

  const auto x_contig = x.contiguous();
  // Calculate the size of the dimension we need to quantize over,
  // For per-channel quant we default to axis 0, since it is only for
  // weight quantization currently.
  int64_t size = 1;

  if (per_row_fq) {
    at::Tensor y = x;
    if (x.dim() != 2) {
      auto res = DimVector(x.sizes());
      std::iota(res.begin(), res.end(), 0);
      res[ch_axis] = 0;
      res[0] = ch_axis;

      y = x.permute(res);
      y = y.flatten(1);
    }
    size = x.size(ch_axis);
    if (running_min.numel() == 0) {
      running_min.resize_(size).fill_(at::numeric_limits<float>::upper_bound());
      running_max.resize_(size).fill_(at::numeric_limits<float>::lower_bound());
      scale.resize_(size);
      zero_point.resize_(size);
    }
    native::xpu::_calculate_moving_average(
        y,
        observer_on,
        running_min,
        running_max,
        averaging_const,
        size,
        per_row_fq);
  } else {
    native::xpu::_calculate_moving_average(
        x_contig,
        observer_on,
        running_min,
        running_max,
        averaging_const,
        size,
        per_row_fq);
  }

  float* scale_ptr = scale.data_ptr<float>();
  int32_t* zp_ptr = zero_point.data_ptr<int32_t>();

  native::xpu::_calc_moving_avg_qparams_helper(
      x_contig,
      fake_quant_on,
      running_min,
      running_max,
      scale_ptr,
      zp_ptr,
      quant_min,
      quant_max,
      symmetric_quant,
      size,
      per_row_fq);

  if (per_row_fq) {
    if (fake_quant_on.item().toInt()) {
      return at::fake_quantize_per_channel_affine_cachemask(
          x, scale, zero_point, 0, quant_min, quant_max);
    } else {
      auto mask = at::ones_like(x, at::kBool, MemoryFormat::Preserve);
      return std::make_tuple(x.clone(), mask);
    }
  } else {
    return at::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
        x, scale, zero_point, fake_quant_on, quant_min, quant_max);
  }
}

} // namespace native
} // namespace at
