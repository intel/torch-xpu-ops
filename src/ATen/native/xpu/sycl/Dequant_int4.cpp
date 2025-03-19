#include <ATen/native/xpu/sycl/Dequant_int4.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {
template <
    typename scalar_t = sycl::half,
    int blocksize = 32,
    int TileK = 1,
    int TileN = 16,
    int SgSize = 16>
struct DequantInt4KernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  DequantInt4KernelFunctor(
      int n,
      int k,
      const uint8_t* weight_int4,
      const scalar_t* ScaleAndZeros,
      scalar_t* weight_dequant)
      : n(n),
        k(k),
        weight_int4(weight_int4),
        ScaleAndZeros(ScaleAndZeros),
        weight_dequant(weight_dequant) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {}
  [[intel::reqd_sub_group_size(SgSize)]] void operator()(
      sycl::nd_item<1> it) const {
    int constexpr GroupN = TileN;
    int constexpr GroupK = SgSize * TileK;
    assert(blocksize % TileK == 0);
    static_assert(TileN == SgSize);
    static_assert(TileK == 1);
    int nsg_k = k / GroupK;
    int nsg_n = n / GroupN;

    int g_idx = it.get_group(0);
    auto sg = it.get_sub_group();
    int sg_id = sg.get_local_id()[0];
    int g_idx_n = g_idx / nsg_k;
    int g_idx_k = g_idx % nsg_k;
    int g_n = g_idx_n * GroupN;
    int g_k = g_idx_k * GroupK;

    int ld_scale_zp = n * 2;
    auto sptr = ScaleAndZeros + g_n * 2 + (g_k / blocksize) * ld_scale_zp;
    auto zptr = ScaleAndZeros + g_n * 2 + (g_k / blocksize) * ld_scale_zp + 1;

    auto bptr = weight_int4 + (g_k + g_n * k) / 2;
    auto dbptr = weight_dequant + g_k * n + g_n;

    float tmp[TileN];
    bool high4 = sg_id % 2 != 0;
    for (int in = 0; in < TileN; in++) {
      int scale_offset = in * 2 + sg_id * TileK / blocksize * ld_scale_zp;
      int zp_offset = scale_offset;
      float scale = *(sptr + scale_offset);
      float zero_point = *(zptr + zp_offset);
      uint8_t srcu8 = *(bptr + (sg_id * TileK + in * k) / 2);
      tmp[in] = high4
          ? static_cast<int8_t>((srcu8 >> 4) - 8) * scale + zero_point
          : static_cast<int8_t>((srcu8 & 0x0f) - 8) * scale + zero_point;
    }

    float tmpT[TileN];

    for (int in = 0; in < TileN; in++) {
      for (int is = 0; is < SgSize; is++) {
        auto shlv = select_from_group(sg, tmp[in], is);
        if (sg_id == in) {
          tmpT[is] = shlv;
        }
      }
    }
    // weight(int4)(col_major) -> weight(dequant)(row_major)
    for (int in = 0; in < TileN; in++) {
      dbptr[sg_id + in * n] = tmpT[in];
    }
  }

 private:
  int n;
  int k;
  const uint8_t* weight_int4;
  const scalar_t* ScaleAndZeros;
  scalar_t* weight_dequant;
};

void dequant_int4_kernel(
    const Tensor& weight_int4,
    Tensor& weight,
    int qGroupSize,
    const Tensor& qScaleAndZeros) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();

  int constexpr SgSize = 16;
  int constexpr TileK = 1;
  int constexpr TileN = 16;
  int constexpr GroupN = TileN;
  int constexpr GroupK = SgSize * TileK;
  assert(qGroupSize % TileK == 0);
  static_assert(TileN == SgSize);
  static_assert(TileK == 1);
  int k = weight.size(0);
  int n = weight.size(1);
  assert(k % GroupK == 0 && n % GroupN == 0);
  int nsg_k = k / GroupK;
  int nsg_n = n / GroupN;
  sycl::range<1> global_range{static_cast<size_t>(nsg_n) * nsg_k * SgSize};
  sycl::range<1> local_range{SgSize};
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      weight.scalar_type(), "dequant_int4_kernel", [&]() {
        using scalar_sycl_t = std::conditional_t<
            std::is_same_v<scalar_t, at::Half>,
            sycl::half,
            sycl::ext::oneapi::bfloat16>;
        switch (qGroupSize) {
          case 16: {
            auto kfn = DequantInt4KernelFunctor<
                scalar_sycl_t,
                16,
                TileK,
                TileN,
                SgSize>(
                n,
                k,
                reinterpret_cast<const uint8_t*>(weight_int4.data_ptr()),
                reinterpret_cast<const scalar_sycl_t*>(
                    qScaleAndZeros.data_ptr<scalar_t>()),
                reinterpret_cast<scalar_sycl_t*>(weight.data_ptr<scalar_t>()));
            sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
            break;
          }
          case 32: {
            auto kfn = DequantInt4KernelFunctor<
                scalar_sycl_t,
                32,
                TileK,
                TileN,
                SgSize>(
                n,
                k,
                reinterpret_cast<const uint8_t*>(weight_int4.data_ptr()),
                reinterpret_cast<const scalar_sycl_t*>(
                    qScaleAndZeros.data_ptr<scalar_t>()),
                reinterpret_cast<scalar_sycl_t*>(weight.data_ptr<scalar_t>()));
            sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
            break;
          }
          case 64: {
            auto kfn = DequantInt4KernelFunctor<
                scalar_sycl_t,
                64,
                TileK,
                TileN,
                SgSize>(
                n,
                k,
                reinterpret_cast<const uint8_t*>(weight_int4.data_ptr()),
                reinterpret_cast<const scalar_sycl_t*>(
                    qScaleAndZeros.data_ptr<scalar_t>()),
                reinterpret_cast<scalar_sycl_t*>(weight.data_ptr<scalar_t>()));
            sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
            break;
          }
          case 128: {
            auto kfn = DequantInt4KernelFunctor<
                scalar_sycl_t,
                128,
                TileK,
                TileN,
                SgSize>(
                n,
                k,
                reinterpret_cast<const uint8_t*>(weight_int4.data_ptr()),
                reinterpret_cast<const scalar_sycl_t*>(
                    qScaleAndZeros.data_ptr<scalar_t>()),
                reinterpret_cast<scalar_sycl_t*>(weight.data_ptr<scalar_t>()));
            sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
            break;
          }
          case 256: {
            auto kfn = DequantInt4KernelFunctor<
                scalar_sycl_t,
                256,
                TileK,
                TileN,
                SgSize>(
                n,
                k,
                reinterpret_cast<const uint8_t*>(weight_int4.data_ptr()),
                reinterpret_cast<const scalar_sycl_t*>(
                    qScaleAndZeros.data_ptr<scalar_t>()),
                reinterpret_cast<scalar_sycl_t*>(weight.data_ptr<scalar_t>()));
            sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
            break;
          }
        }
      });
}

} // namespace at::native::xpu