#include <ATen/native/xpu/sycl/Dequant_int4.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {
void dequant_int4_kernel(
    const Tensor& weight_int4,
    Tensor& weight,
    int qGroupSize,
    const Tensor& qScaleAndZeros) {
  int constexpr SgSize = 16;
  int constexpr TileK = 16;
  int constexpr TileN = 16;

  int constexpr GroupN = SgSize * TileN;
  int constexpr GroupK = TileK;
  static_assert(TileN % 2 == 0);
  assert(qGroupSize % TileK == 0);
  int n = weight.size(0);
  int k = weight.size(1);
  int nsg_k = k / GroupK;
  int nsg_n = n / GroupN;
  sycl::range<1> global_range{static_cast<size_t>(nsg_n) * nsg_k * SgSize};
  sycl::range<1> local_range{SgSize};
}

template <
    typename scalar_t = sycl::half,
    int blocksize = 32,
    int TileK = 16,
    int TileN = 2,
    int SgSize = 16>
struct DequantInt4KernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  DequantInt4KernelFunctor(
      int n,
      int k,
      const uint8_t* weight_int4,
      const scalar_t* ScaleAndZeros,
      scalar_t* weight_dequant) {
    this->n = n;
    this->k = k;
    this->weight_int4 = weight_int4;
    this->ScaleAndZeros = ScaleAndZeros;
    this->weight_dequant = weight_dequant;
  }
  void sycl_ker_config_convention(sycl::handler& cgh) {}
  [[intel::reqd_sub_group_size(16)]] void operator()(
      sycl::nd_item<1> it) const {
    int constexpr GroupN = SgSize * TileN;
    int constexpr GroupK = TileK;
    static_assert(TileN % 2 == 0);
    assert(blocksize % TileK == 0);
    int nsg_k = k / GroupK;
    int nsg_n = n / GroupN;

    int g_idx = it.get_group(0);
    auto sg = it.get_sub_group();
    int sg_id = sg.get_local_id()[0];
    int g_idx_n = g_idx % nsg_n;
    int g_idx_k = g_idx / nsg_n;
    int g_n = g_idx_n * GroupN;
    int g_k = g_idx_k * GroupK;
    int ldb = k / blocksize;
    auto sptr = ScaleAndZeros + g_k / blocksize * ldb + g_n;
    auto bptr = weight_int4 + (g_k * ldb + g_n) / 2;
    auto dbptr = weight_dequant + g_k * n + g_n;
    float tmp[TileK * TileN];
    float scale[TileN];
    for (int in = 0; in < TileN; in += 1) {
      scale[in] = sptr[sg_id * TileN + in];
    }
    for (int ik = 0; ik < TileK; ik += 1) {
      for (int in = 0; in < TileN; in += 2) {
        uint8_t srcu8 = *(bptr + (ik * ldb + sg_id * TileN + in) / 2);
        tmp[ik * TileN + in] =
            static_cast<int8_t>((srcu8 & 0x0f) - 8) * scale[in];
        tmp[ik * TileN + in + 1] =
            static_cast<int8_t>((srcu8 >> 4) - 8) * scale[in + 1];
      }
    }
    for (int ik = 0; ik < TileK; ik += 1) {
      for (int in = 0; in < TileN; in += 1) {
        dbptr[ik * n + sg_id * TileN + in] = tmp[ik * TileN + in];
      }
    }
  }

 private:
  int n;
  int k;
  const uint8_t* weight_int4;
  const scalar_t* ScaleAndZeros;
  scalar_t* weight_dequant;
};

} // namespace at::native::xpu