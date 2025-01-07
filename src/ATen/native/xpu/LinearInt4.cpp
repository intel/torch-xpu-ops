
#include <ATen/core/op_registration/adaption.h>
#include <ATen/div_rtn.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <ATen/native/xpu/sycl/LinearInt4.h>
#include <comm/xpu_aten.h>

namespace at::native {
Tensor _weight_int4pack_mm_xpu(
    const Tensor& A,
    const Tensor& B,
    int64_t qGroupSize,
    const Tensor& qScaleAndZeros) {
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1);
  TORCH_CHECK(
      A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
      __func__,
      " : expect A to be either 32-bit or 16-bit float tensor.");
  TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
  TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");

  TORCH_CHECK(
      B.dtype() == kInt || B.dtype() == kUInt32 || B.dtype() == kByte,
      __func__,
      " : expect B to be int32 or uint32 or uint8 tensor.");
  TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
  TORCH_CHECK(B.dim() == 2, __func__, " : expect B to 2d tensor.");

  TORCH_CHECK(
      qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||
          qGroupSize == 256,
      __func__,
      ": expect qGroupSize to be 32, 64, 128 or 256, got ",
      qGroupSize);

  TORCH_CHECK(
      qScaleAndZeros.dim() == 3 && qScaleAndZeros.size(0) == N &&
          qScaleAndZeros.size(2) == 2,
      __func__,
      ": expect qScaleAndZeros to be 3d tensor with sizes [",
      N,
      ", :, 2]");

  std::optional<Device> common_device = std::nullopt;
  c10::impl::check_and_update_common_device(
      common_device, A, "xpu::_weight_int4pack_mm", "A");
  c10::impl::check_and_update_common_device(
      common_device, B, "xpu::_weight_int4pack_mm", "B");
  c10::impl::check_and_update_common_device(
      common_device,
      qScaleAndZeros,
      "xpu::_weight_int4pack_mm",
      "qScaleAndZeros");
  Tensor C = at::empty({M, N}, A.options());

  at::native::xpu::linear_int4_kernel(A, B, qGroupSize, qScaleAndZeros, C);
  return C;
}
} // namespace at::native
