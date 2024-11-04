#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/WeightInt4PackKernel.h>

namespace at::native::xpu {

struct WeightToInt4PackKernelFunctor {
  void operator()(sycl::item<1> item) const {
    auto idx = item.get_linear_id();
    int i = idx / K_div_2_;
    int k = idx % K_div_2_;
    int nb_size = std::min(BLOCK_N_, N_ - i * BLOCK_N_);
    uint8_t* src = weight_ + i * BLOCK_N_ * K_div_2_;
    uint8_t* dst = weight_packed_ + i * K_ * BLOCK_N_ / 2;
    for (int n = 0; n < nb_size; n += 2) {
      uint8_t val0 = src[n * K_div_2_ + k];
      uint8_t val1 = src[n * K_div_2_ + K_div_2_ + k];
      uint8_t packed_0 = ((val1 & 0xF0)) | ((val0 & 0xF0) >> 4);
      uint8_t packed_1 = ((val1 & 0xF) << 4) | (val0 & 0xF);
      dst[k * 2 * nb_size / 2 + n / 2] = packed_0;
      dst[(k * 2 + 1) * nb_size / 2 + n / 2] = packed_1;
    }
  }
  WeightToInt4PackKernelFunctor(
      uint8_t* weight_packed,
      uint8_t* weight,
      int K_div_2,
      int BLOCK_N,
      int N,
      int K)
      : weight_packed_(weight_packed),
        weight_(weight),
        K_div_2_(K_div_2),
        BLOCK_N_(BLOCK_N),
        N_(N),
        K_(K) {}

 private:
  uint8_t* weight_packed_;
  uint8_t* weight_;
  int K_div_2_;
  int BLOCK_N_;
  int N_;
  int K_;
};

void weight_to_int4pack_kernel(
    const Tensor& weight_packed,
    const Tensor& weight,
    int N,
    int K) {
  auto weight_packed_data =
      reinterpret_cast<uint8_t*>(weight_packed.data_ptr());
  const auto weight_data = weight.data_ptr<uint8_t>();
  constexpr int BLOCK_N = sizeof(float) * 4; // 16
  const int NB = (N + BLOCK_N - 1) / BLOCK_N;
  int K_div_2 = K / 2;
  size_t global_range = NB * K_div_2;
  auto fn = WeightToInt4PackKernelFunctor(
      weight_packed_data, weight_data, K_div_2, BLOCK_N, N, K);
  sycl_kernel_submit(sycl::range<1>(global_range), getCurrentSYCLQueue(), fn);
}

} // namespace at::native::xpu
