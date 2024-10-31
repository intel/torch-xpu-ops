#include <ATen/native/xpu/sycl/WeightInt4PackKernel.h>

namespace at::native {

Tensor _convert_weight_to_int4pack_xpu(const Tensor& in, int64_t innerKTiles) {
  TORCH_CHECK(in.dim() == 2, __func__, " : expect weight to be 2D tensor.");
  TORCH_CHECK(in.dtype() == at::kInt, __func__, " : expect weight to be kInt.");
  TORCH_CHECK(
      innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8,
      __func__,
      " : innerKTiles need to be 2, 4, or 8, got ",
      innerKTiles);

  auto weight = in.contiguous();
  auto N = weight.size(0);
  auto K = weight.size(1);

  // Create fake shapes for cpu. The meta registration in dynamo requires
  // operator has the same output shape for each device. So creating a fake
  // shape {N / 8, K / (16 * innerKTiles), 32, innerKTiles / 2}
  constexpr int64_t kNTileSize = 8;
  constexpr int64_t kKTileSize = 16;
  auto nTiles = (N + kNTileSize - 1) / kNTileSize;

  TORCH_CHECK(N % 16 == 0, __func__, " : expect N to be dividable by 16");
  const int64_t kSuperKTileSize = kKTileSize * innerKTiles;
  TORCH_CHECK(
      K % kSuperKTileSize == 0,
      __func__,
      " : epxect K to be dividable by ",
      kSuperKTileSize);
  auto kSuperTiles = (K + kSuperKTileSize - 1) / kSuperKTileSize;

  auto weight_packed = at::empty(
      {nTiles, kSuperTiles, 32, innerKTiles / 2},
      at::TensorOptions().dtype(at::kInt).device(in.device()));

  native::xpu::weight_to_int4pack_kernel(weight_packed, weight, N, K, 2);
  return weight_packed;
}

} // namespace at::native