
#include <ATen/core/op_registration/adaption.h>
#include <ATen/div_rtn.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <ATen/native/xpu/sycl/LinearInt4.h>
#include <comm/xpu_aten.h>

namespace at::native {
Tensor& _weight_int4pack_mm_with_scales_and_zeros_xpu(
    const Tensor& input,
    const Tensor& weight,
    int qGroupSize,
    const Tensor& weight_scale_zero_point) {
  auto M = input.size(0);
  auto N = weight.size(0);
  auto K = input.size(1);
  TORCH_CHECK(
      input.dtype() == kBFloat16 || input.dtype() == kHalf ||
          input.dtype() == kFloat,
      __func__,
      " : expect input to be either 32-bit or 16-bit float tensor.");

  TORCH_CHECK(
      weight.dtype() == kByte, __func__, " : expect B to be uint8 tensor.");
  TORCH_CHECK(
      weight.is_contiguous(), __func__, " : expect B to be contiguous.");
  TORCH_CHECK(
      weight.size(1) == K / 2,
      __func__,
      " : expect B.size(1) to be K/2, got ",
      weight.size(1));

  TORCH_CHECK(
      qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
          qGroupSize == 256,
      __func__,
      ": expect qGroupSize to be 32, 64, 128 or 256, got ",
      qGroupSize);

  TORCH_CHECK(
      weight_scale_zero_point.dim() == 3 &&
          weight_scale_zero_point.size(1) == N &&
          weight_scale_zero_point.size(2) == 2,
      __func__,
      ": expect weight_scale_zero_point to be 3d tensor with sizes [:, ",
      N,
      ", 2]");

  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, input, "xpu::linear_int4", "input");
  c10::impl::check_and_update_common_device(
      common_device, weight, "xpu::linear_int4", "weight");
  c10::impl::check_and_update_common_device(
      common_device,
      weight_scale_zero_point,
      "xpu::linear_int4",
      "weight_scale_zero_point");
  Tensor output = at::empty({M, N}, input.options());

  at::native::xpu::linear_int4_kernel(
      input, weight, qGroupSize, weight_scale_zero_point, output);
  return output;
}
} // namespace at::native
