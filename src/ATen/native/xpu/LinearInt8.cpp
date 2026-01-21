#include <ATen/core/op_registration/adaption.h>
#include <ATen/div_rtn.h>
#include <ATen/native/TensorIterator.h>
#include <torch/library.h>

#include <ATen/native/xpu/sycl/LinearInt8.h>
#include <comm/xpu_aten.h>

namespace at::native {
    Tensor _weight_int8pack_mm_xpu(
        const Tensor& A,
        const Tensor& B,
        const Tensor& scales) {
      auto M = A.size(0);
      auto N = B.size(0);
      auto K = A.size(1);
      TORCH_CHECK(
          A.dtype() == kBFloat16 || A.dtype() == kHalf || A.dtype() == kFloat,
          __func__,
          " : expect A to be either 32-bit or 16-bit float tensor.");
      TORCH_CHECK(A.is_contiguous(), __func__, " : expect A to be contiguous.");
      TORCH_CHECK(A.dim() == 2, __func__, " : expect A to be 2D tensor.");
    
      TORCH_CHECK(B.dtype() == kChar, __func__, " : expect B to be int8 tensor.");
      TORCH_CHECK(B.is_contiguous(), __func__, " : expect B to be contiguous.");
      TORCH_CHECK(B.dim() == 2, __func__, " : expect B to 2d tensor.");
      TORCH_CHECK(B.size(1) == K, __func__, " : expect B.size(1) == ", K);

      TORCH_CHECK(scales.dim() == 1 && scales.size(0) == N,
      __func__, " : expect scales to be 1d tensor with size ", N);
    
      std::optional<Device> common_device = std::nullopt;
      c10::impl::check_and_update_common_device(
          common_device, A, "xpu::_weight_int8pack_mm", "A");
      c10::impl::check_and_update_common_device(
          common_device, B, "xpu::_weight_int8pack_mm", "B");
      c10::impl::check_and_update_common_device(
          common_device,
          scales,
          "xpu::_weight_int8pack_mm",
          "scales");
      Tensor C = at::empty({M, N}, A.options());
      // When M > 1 will use two kernels(dequant and gemm)
      // When M == 1 will use one linear_int4_kernel(dequant and gemv)
      // if (M > 1) {
      //   Tensor B_dequant = at::empty({K, N}, A.options());
      //   at::native::xpu::dequant_int4_kernel(
      //       B, B_dequant, qGroupSize, qScaleAndZeros);
      //   C = A.matmul(B_dequant);
      // } else {
      //   at::native::xpu::linear_int4_kernel(A, B, qGroupSize, qScaleAndZeros, C);
      // }
      at::native::xpu::linear_int8_kernel(A, B, scales, C);
      return C;
    }
}