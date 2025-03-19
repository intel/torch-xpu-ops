#include <ATen/native/xpu/sycl/GroupReduceUtils.h>
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

template <typename scalar_t = at::BFloat16, int block_size = 32>
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
    int ld_scale_zp = 2 * n;
    if (k % (SgSize * 32 * Unroll) == 0) {
      int constexpr TileK = 32;
      int constexpr GroupK = SgSize * TileK;

      int g_idx = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      int g_n = g_idx;
      auto sptr = ScaleAndZeros + g_n * 2;
      auto zptr = ScaleAndZeros + g_n * 2 + 1;
      auto bptr = B + g_n * k / 2;
      auto aptr = A;
      auto cptr = C + g_n;

      float tmpAcc = 0.f;
      for (int i = 0; i < k; i += GroupK * Unroll) {
#pragma unroll
        for (int iu = 0; iu < Unroll; iu++) {
          const uint8_t* tmps8 =
              reinterpret_cast<const uint8_t*>(bptr + sg_id * TileK / 2);
          int scale_offset = sg_id * (TileK / blocksize) * ld_scale_zp;
          int zp_offset = sg_id * (TileK / blocksize) * ld_scale_zp;
          scalar_t scale = *(sptr + scale_offset);
          scalar_t zero_point = *(zptr + zp_offset);
#pragma unroll
          for (int ikk = 0; ikk < TileK; ikk += 2) {
            scalar_t tmpA0 = *(aptr + sg_id * TileK + ikk);
            scalar_t tmpA1 = *(aptr + sg_id * TileK + ikk + 1);
            scalar_t tmpB0 = static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8);
            scalar_t tmpB1 = static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8);
            scalar_t tmpAmulB0 = tmpA0 * (tmpB0 * scale + zero_point);
            scalar_t tmpAmulB1 = tmpA1 * (tmpB1 * scale + zero_point);
            tmpAcc += static_cast<float>(tmpAmulB0);
            tmpAcc += static_cast<float>(tmpAmulB1);
          }
          sptr += (GroupK / blocksize) * ld_scale_zp;
          zptr += (GroupK / blocksize) * ld_scale_zp;
          aptr += GroupK;
          bptr += GroupK / 2;
        }
      }
      float sum = 0.f;
      sum += SubgroupReduceSumWithoutBroadcast<float, 16>(it, tmpAcc);
      if (sg_id == 0) {
        *cptr = static_cast<scalar_t>(sum);
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
      auto sptr = ScaleAndZeros + g_n * 2;
      auto zptr = ScaleAndZeros + g_n * 2 + 1;
      auto bptr = B + g_n * k / 2;
      auto aptr = A;
      auto cptr = C + g_n;
      float tmpAcc = 0.f;
      int i = 0;
      for (; i < k_body; i += GroupK * Unroll) {
#pragma unroll
        for (int iu = 0; iu < Unroll; iu++) {
          const uint8_t* tmps8 =
              reinterpret_cast<const uint8_t*>(bptr + sg_id * TileK / 2);
          int scale_offset = sg_id * TileK / blocksize * ld_scale_zp;
          int zp_offset = sg_id * TileK / blocksize * ld_scale_zp;

          scalar_t scale = *(sptr + scale_offset);
          scalar_t zero_point = *(zptr + zp_offset);
#pragma unroll
          for (int ikk = 0; ikk < TileK; ikk += 2) {
            scalar_t tmpA0 = *(aptr + sg_id * TileK + ikk);
            scalar_t tmpA1 = *(aptr + sg_id * TileK + ikk + 1);
            scalar_t tmpB0 = static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8);
            scalar_t tmpB1 = static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8);
            scalar_t tmpAmulB0 = tmpA0 * (tmpB0 * scale + zero_point);
            scalar_t tmpAmulB1 = tmpA1 * (tmpB1 * scale + zero_point);
            tmpAcc += static_cast<float>(tmpAmulB0);
            tmpAcc += static_cast<float>(tmpAmulB1);
          }
          sptr += (GroupK / blocksize) * ld_scale_zp;
          zptr += (GroupK / blocksize) * ld_scale_zp;
          aptr += GroupK;
          bptr += GroupK / 2;
        }
      }
      if (i + GroupK2 * Unroll < k_body2) {
        for (; i < k_body2; i += GroupK2 * Unroll) {
#pragma unroll
          for (int iu = 0; iu < Unroll; iu++) {
            const uint8_t* tmps8 =
                reinterpret_cast<const uint8_t*>(bptr + sg_id * TileK2 / 2);
            int scale_offset = sg_id * TileK2 / blocksize * ld_scale_zp;
            int zp_offset = sg_id * TileK2 / blocksize * ld_scale_zp;
            scalar_t scale = *(sptr + scale_offset);
            scalar_t zero_point = *(zptr + zp_offset);
#pragma unroll
            for (int ikk = 0; ikk < TileK2; ikk += 2) {
              scalar_t tmpA0 = *(aptr + sg_id * TileK2 + ikk);
              scalar_t tmpA1 = *(aptr + sg_id * TileK2 + ikk + 1);
              scalar_t tmpB0 = static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8);
              scalar_t tmpB1 = static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8);
              scalar_t tmpAmulB0 = tmpA0 * (tmpB0 * scale + zero_point);
              scalar_t tmpAmulB1 = tmpA1 * (tmpB1 * scale + zero_point);
              tmpAcc += static_cast<float>(tmpAmulB0);
              tmpAcc += static_cast<float>(tmpAmulB1);
            }
            sptr += (GroupK2 / blocksize) * ld_scale_zp;
            zptr += (GroupK2 / blocksize) * ld_scale_zp;
            aptr += GroupK2;
            bptr += GroupK2 / 2;
          }
        }
      }
      if (i + SgSize * 2 <= k) {
        for (; i < k; i += SgSize * 2) {
          uint8_t tmps8 = *(bptr + sg_id);

          int scale_offset = sg_id * 2 / blocksize * ld_scale_zp;
          int zp_offset = sg_id * 2 / blocksize * ld_scale_zp;
          scalar_t scale = *(sptr + scale_offset);
          scalar_t zero_point = *(zptr + zp_offset);

          scalar_t tmpA0 = *(aptr + sg_id * 2);
          scalar_t tmpA1 = *(aptr + sg_id * 2 + 1);
          scalar_t tmpB0 = static_cast<int8_t>((tmps8 & 0x0f) - 8);
          scalar_t tmpB1 = static_cast<int8_t>((tmps8 >> 4) - 8);

          scalar_t tmpAmulB0 = tmpA0 * (tmpB0 * scale + zero_point);
          scalar_t tmpAmulB1 = tmpA1 * (tmpB1 * scale + zero_point);
          tmpAcc += static_cast<float>(tmpAmulB0);
          tmpAcc += static_cast<float>(tmpAmulB1);
          sptr += (SgSize * 2 / blocksize) * ld_scale_zp;
          zptr += (SgSize * 2 / blocksize) * ld_scale_zp;
          aptr += SgSize * 2;
          bptr += SgSize * 2 / 2;
        }
      }
      float sum = 0.f;
      sum += SubgroupReduceSumWithoutBroadcast<float, 16>(it, tmpAcc);
      if (sg_id == 0) {
        *cptr = static_cast<scalar_t>(sum);
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
        const scalar_t* input_data = A.const_data_ptr<scalar_t>();
        const uint8_t* weight_data = reinterpret_cast<const uint8_t*>(
            B.const_data_ptr()); // int4x2 or int4x8

        scalar_t* output_data = C.mutable_data_ptr<scalar_t>();
        const scalar_t* scale_zeros_data =
            qScaleAndZeros.const_data_ptr<scalar_t>();

        switch (qGroupSize) {
          case 16: {
            auto kfn = LinearInt4KernelFunctor<scalar_t, 16>(
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
            break;
          }
          case 32: {
            auto kfn = LinearInt4KernelFunctor<scalar_t, 32>(
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
            break;
          }
          case 64: {
            auto kfn = LinearInt4KernelFunctor<scalar_t, 64>(
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
            break;
          }
          case 128: {
            auto kfn = LinearInt4KernelFunctor<scalar_t, 128>(
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
            break;
          }
          case 256: {
            auto kfn = LinearInt4KernelFunctor<scalar_t, 256>(
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
            break;
          }
        }
      });
}

} // namespace at::native::xpu