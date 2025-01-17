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
      scalar_t* weight_dequant,
      sycl::stream os)
      : n(n),
        k(k),
        weight_int4(weight_int4),
        ScaleAndZeros(ScaleAndZeros),
        weight_dequant(weight_dequant),
        os_(os) {}

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

    auto sptr = ScaleAndZeros + (g_k / blocksize + g_n * (k / blocksize)) * 2;
    auto zptr = ScaleAndZeros + (g_k / blocksize + g_n * (k / blocksize)) * 2 + 1;

    auto bptr = weight_int4 + (g_k + g_n * k) / 2;
    auto dbptr = weight_dequant + g_k * n + g_n;

    float tmp[TileN];
    bool high4 = sg_id % 2 != 0;
    for (int in = 0; in < TileN; in++) {
      int scale_offset = sg_id * (TileK / blocksize) * 2 + in * (k / blocksize) * 2;
      int zp_offset = sg_id * (TileK / blocksize) * 2 + in * (k / blocksize) * 2;
      // float scale = sptr[sg_id * TileK / blocksize + in * ldb];
      float scale = *(sptr + scale_offset);
      float zero_point = *(zptr + zp_offset);

      uint8_t srcu8 = *(bptr + (sg_id * TileK + in * k) / 2);
      tmp[in] = high4
          ? static_cast<int8_t>((srcu8 >> 4) - 8) * scale + zero_point
          : static_cast<int8_t>((srcu8 & 0x0f) - 8) * scale + zero_point;
      if (g_idx == 0) {
        os_ << "sg: " << sg_id << " " << " fp16 " << (float)tmp[in] << " zepo "
            << zero_point << " scale " << scale << " srcu8 "
            << (high4 ? static_cast<int8_t>((srcu8 >> 4))
                      : static_cast<int8_t>((srcu8 & 0x0f)))
            << " tmp " << tmp[in] << sycl::endl;
      }
    }

    float tmpT[TileN];
    for (int in = 0; in < TileN; in++) {
      for (int is = 0; is < SgSize; is++) {
        auto shlv = group_broadcast(sg, tmp[in], is);
        if (sg_id == in) {
          tmpT[is] = shlv;
        }
      }
    }
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
  sycl::stream os_;
};

void dequant_int4_kernel(
    const Tensor& weight_int4,
    Tensor& weight,
    int qGroupSize,
    const Tensor& qScaleAndZeros) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  using scalar_t = sycl::half;

  int constexpr SgSize = 16;
  int constexpr TileK = 1;
  int constexpr TileN = 16;
  int constexpr GroupN = TileN;
  int constexpr GroupK = SgSize * TileK;
  assert(qGroupSize % TileK == 0);
  static_assert(TileN == SgSize);
  static_assert(TileK == 1);
  int n = weight.size(0);
  int k = weight.size(1);
  int nsg_k = k / GroupK;
  int nsg_n = n / GroupN;
  sycl::range<1> global_range{static_cast<size_t>(nsg_n) * nsg_k * SgSize};
  sycl::range<1> local_range{SgSize};
  // DequantInt4KernelFunctor<scalar_t, 32, TileK, TileN, SgSize> kfn =
  //     DequantInt4KernelFunctor<scalar_t, 32, TileK, TileN, SgSize>(
  //         n,
  //         k,
  //         reinterpret_cast<const uint8_t*>(weight_int4.data_ptr()),
  //         reinterpret_cast<const scalar_t*>(qScaleAndZeros.data_ptr()),
  //         reinterpret_cast<scalar_t*>(weight.data_ptr()));
  // sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);

  auto cgf = [&](::sycl::handler& cgh) {
    sycl::stream os(1024, 256, cgh);
    DequantInt4KernelFunctor<scalar_t, 32, TileK, TileN, SgSize> kfn =
        DequantInt4KernelFunctor<scalar_t, 32, TileK, TileN, SgSize>(
            n,
            k,
            reinterpret_cast<const uint8_t*>(weight_int4.data_ptr()),
            reinterpret_cast<const scalar_t*>(qScaleAndZeros.data_ptr()),
            reinterpret_cast<scalar_t*>(weight.data_ptr()),
            os);
    kfn.sycl_ker_config_convention(cgh);
    cgh.parallel_for<
        DequantInt4KernelFunctor<scalar_t, 32, TileK, TileN, SgSize>>(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(global_range), ::sycl::range<1>(local_range)),
        kfn);
  };
  sycl_queue.submit(cgf);
}

} // namespace at::native::xpu