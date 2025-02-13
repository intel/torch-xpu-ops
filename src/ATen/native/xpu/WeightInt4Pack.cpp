#include <ATen/native/xpu/sycl/WeightInt4PackKernel.h>

namespace at::native {

// input is [n][k / 2] (uint8 dtype)
// output is [n][k // 8]
Tensor _convert_weight_to_int4pack_xpu(const Tensor& in, int64_t innerKTiles) {
  TORCH_CHECK(in.dim() == 2, __func__, " : expect weight to be 2D tensor.");
  TORCH_CHECK(
      in.dtype() == at::kByte, __func__, " : expect weight to be kByte.");
  TORCH_CHECK(
      innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8,
      __func__,
      " : innerKTiles need to be 2, 4, or 8, got ",
      innerKTiles);

  auto weight = in.contiguous();
  auto N = weight.size(0);
  auto K = weight.size(1) * 2;
  TORCH_CHECK(K % 8 == 0, "The K dimension of int4 GEMM should be a multiple of 8.");
  auto weight_packed = at::empty({N, K / 8}, at::TensorOptions().dtype(at::kInt).device(in.device()));

  xpu::weight_to_int4pack_kernel(weight_packed, weight, N, K);
  return weight_packed;
}

} // namespace at::native
