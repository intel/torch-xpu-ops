#include <ATen/native/xpu/sycl/LinearInt4.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {
static inline int padto_le(int src, int padding) {
  return src / padding * padding;
}

static inline int64_t padto_le(int64_t src, int64_t padding) {
  return src / padding * padding;
}

static inline size_t padto_le(size_t src, int padding) {
  return src / size_t(padding) * size_t(padding);
}

template <typename scalar_t = sycl::half, int block_size = 32>
struct LinearInt4KernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  LinearInt4KernelFunctor(
      const scalar_t* A,
      const uint8_t* B,
      scalar_t* C,
      const scalar_t* ScaleAndZeros,
      int m,
      int n,
      int k,
      int lda,
      int ldb,
      int ldc)
      : A(A),
        B(B),
        C(C),
        ScaleAndZeros(ScaleAndZeros),
        m(m),
        n(n),
        k(k),
        lda(lda),
        ldb(ldb),
        ldc(ldc) {}
  void sycl_ker_config_convention(sycl::handler& cgh) {}

  [[intel::reqd_sub_group_size(16)]] void operator()(
      sycl::nd_item<1> it) const {
    int constexpr Unroll = 2;
    int constexpr SgSize = 16;
    int constexpr blocksize = block_size;
    using scalarx2_t = sycl::vec<scalar_t, 2>;

    if (k % (SgSize * 32 * Unroll) == 0) {
      int constexpr TileK = 32;
      int constexpr GroupK = SgSize * TileK;

      int g_idx = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      int g_n = g_idx;
      auto sptr = ScaleAndZeros + g_n * ldb * 2;
      auto zptr = ScaleAndZeros + g_n * ldb * 2 + 1;
      auto bptr = B + g_n * k / 2;
      auto aptr = A;
      auto cptr = C + g_n;

      sycl::float2 tmpAcc = {0.f, 0.f};
      for (int i = 0; i < k; i += GroupK * Unroll) {
#pragma unroll
        for (int iu = 0; iu < Unroll; iu++) {
          uint8_t tmps8[TileK / 2];
          *(sycl::vec<uint8_t, TileK / 2>*)tmps8 =
              *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
          int scale_offset = sg_id * (TileK / blocksize) * 2;
          int zp_offset = sg_id * (TileK / blocksize) * 2;
          scalar_t scale = *(sptr + scale_offset);
          scalar_t zero_point = *(zptr + zp_offset);
#pragma unroll
          for (int ikk = 0; ikk < TileK; ikk += 2) {
            scalarx2_t tmpA = *(scalarx2_t*)(aptr + sg_id * TileK + ikk);
            scalarx2_t tmpB = {
                static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8),
                static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8)};
            auto tmpAmulB = tmpA * (tmpB * scale + zero_point);
            tmpAcc += {tmpAmulB[0], tmpAmulB[1]};
          }
          sptr += (GroupK / blocksize) * 2;
          aptr += GroupK;
          bptr += GroupK / 2;
        }
      }
      sycl::float2 sum = {0.f, 0.f};
      sum += sycl::reduce_over_group(sg, tmpAcc, sycl::plus<>());
      if (sg_id == 0) {
        *cptr = static_cast<scalar_t>(sum[0] + sum[1]);
      }
    } else { // k % (SgSize * 32 * Unroll) != 0
      int constexpr TileK = 32;
      int constexpr GroupK = SgSize * TileK;
      int k_body = padto_le(k, GroupK * Unroll);
      int constexpr TileK2 = 8;
      int constexpr GroupK2 = SgSize * TileK2;
      int k_body2 = padto_le(k, GroupK2 * Unroll);
      int g_idx = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      int g_n = g_idx;
      auto sptr = ScaleAndZeros + g_n * ldb * 2;
      auto zptr = ScaleAndZeros + g_n * ldb * 2 + 1;
      auto bptr = B + g_n * k / 2;
      auto aptr = A;
      auto cptr = C + g_n;
      sycl::float2 tmpAcc = {0.f, 0.f};
      int i = 0;
      for (; i < k_body; i += GroupK * Unroll) {
#pragma unroll
        for (int iu = 0; iu < Unroll; iu++) {
          uint8_t tmps8[TileK / 2];
          *(sycl::vec<uint8_t, TileK / 2>*)tmps8 =
              *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);

          int scale_offset = sg_id * (TileK / blocksize) * 2;
          int zp_offset = sg_id * (TileK / blocksize) * 2;
          scalar_t scale = *(sptr + scale_offset);
          scalar_t zero_point = *(zptr + zp_offset);
#pragma unroll
          for (int ikk = 0; ikk < TileK; ikk += 2) {
            scalarx2_t tmpA = *(scalarx2_t*)(aptr + sg_id * TileK + ikk);
            scalarx2_t tmpB = {
                static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8),
                static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8)};
            auto tmpAmulB = tmpA * (tmpB * scale + zero_point);
            tmpAcc += {tmpAmulB[0], tmpAmulB[1]};
          }
          sptr += (GroupK / blocksize) * 2;
          aptr += GroupK;
          bptr += GroupK / 2;
        }
      }
      if (i + GroupK2 * Unroll < k_body2) {
        for (; i < k_body2; i += GroupK2 * Unroll) {
#pragma unroll
          for (int iu = 0; iu < Unroll; iu++) {
            uint8_t tmps8[TileK2 / 2];
            *(sycl::vec<uint8_t, TileK2 / 2>*)tmps8 =
                *(sycl::vec<uint8_t, TileK2 / 2>*)(bptr + sg_id * TileK2 / 2);

            int scale_offset = sg_id * (TileK2 / blocksize) * 2;
            int zp_offset = sg_id * (TileK2 / blocksize) * 2;
            scalar_t scale = *(sptr + scale_offset);
            scalar_t zero_point = *(zptr + zp_offset);
#pragma unroll
            for (int ikk = 0; ikk < TileK2; ikk += 2) {
              scalarx2_t tmpA = *(scalarx2_t*)(aptr + sg_id * TileK2 + ikk);
              scalarx2_t tmpB = {
                  static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8),
                  static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8)};
              auto tmpAmulB = tmpA * (tmpB * scale + zero_point);
              tmpAcc += {tmpAmulB[0], tmpAmulB[1]};
            }
            sptr += (GroupK2 / blocksize) * 2;
            aptr += GroupK2;
            bptr += GroupK2 / 2;
          }
        }
      }
      if (i + SgSize * 2 <= k) {
        for (; i < k; i += SgSize * 2) {
          uint8_t tmps8 = *(bptr + sg_id);
          scalarx2_t tmpB = {
              static_cast<int8_t>((tmps8 & 0x0f) - 8),
              static_cast<int8_t>((tmps8 >> 4) - 8)};

          int scale_offset = sg_id * (2 / blocksize) * 2;
          int zp_offset = sg_id * (2 / blocksize) * 2;
          scalar_t scale = *(sptr + scale_offset);
          scalar_t zero_point = *(zptr + zp_offset);
          scalarx2_t tmpA = *(scalarx2_t*)(aptr + sg_id * 2);
          auto tmpAmulB = tmpA * (tmpB * scale + zero_point);
          tmpAcc += {tmpAmulB[0], tmpAmulB[1]};
          sptr += (SgSize * 2 / blocksize) * 2;
          aptr += SgSize * 2;
          bptr += SgSize * 2 / 2;
        }
      }
      sycl::float2 sum = {0.f, 0.f};
      sum += sycl::reduce_over_group(sg, tmpAcc, sycl::plus<>());
      if (sg_id == 0) {
        *cptr = static_cast<scalar_t>(sum[0] + sum[1]);
      }
    }
  }

 private:
  const scalar_t* A;
  const uint8_t* B;
  scalar_t* C;
  const scalar_t* ScaleAndZeros;
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
};

void linear_int4_kernel(
    const Tensor& A,
    const Tensor& B,
    int qGroupSize,
    const Tensor& qScaleAndZeros,
    Tensor& C) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  int64_t m = A.size(0);
  int64_t n = C.size(1);
  int64_t k = A.size(1);
  int constexpr SgSize = 16;
  sycl::range<1> local_range{SgSize};
  sycl::range<1> global_range{static_cast<size_t>(n) * SgSize};
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      A.scalar_type(), "linear_int4_kernel", [&]() {
        using scalar_sycl_t = std::conditional_t<
            std::is_same_v<scalar_t, at::Half>,
            sycl::half,
            sycl::ext::oneapi::bfloat16>;

        const scalar_sycl_t* input_data =
            reinterpret_cast<scalar_sycl_t*>(A.data_ptr<scalar_t>());
        uint8_t* weight_data =
            reinterpret_cast<uint8_t*>(B.data_ptr()); // int4x2 or int4x8

        scalar_sycl_t* output_data =
            reinterpret_cast<scalar_sycl_t*>(C.data_ptr<scalar_t>());
        scalar_sycl_t* scale_zeros_data = reinterpret_cast<scalar_sycl_t*>(
            qScaleAndZeros.data_ptr<scalar_t>());
        LinearInt4KernelFunctor<scalar_sycl_t, 32> kfn =
            LinearInt4KernelFunctor<scalar_sycl_t, 32>(
                input_data,
                weight_data,
                output_data,
                scale_zeros_data,
                m,
                n,
                k,
                k,
                k / qGroupSize,
                n);
        sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
      });
}

} // namespace at::native::xpu