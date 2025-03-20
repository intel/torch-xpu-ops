#include <ATen/native/xpu/sycl/LinearInt8.h>
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

template <typename scalar_t = sycl::half>
struct LinearInt8KernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  LinearInt8KernelFunctor(
      const scalar_t* A,
      const uint8_t* B,
      scalar_t* C,
      const scalar_t* scales,
      int m,
      int n,
      int k,
      int lda,
      int ldb,
      int ldc)
      : A(A),
        B(B),
        C(C),
        scales(scales),
        m(m),
        n(n),
        k(k),
        lda(lda),
        ldb(ldb),
        ldc(ldc) {}
  void sycl_ker_config_convention(sycl::handler& cgh) {}

  [[intel::reqd_sub_group_size(16)]] void operator()(
      sycl::nd_item<1> it) const {
    int constexpr Unroll = 1;
    int constexpr SgSize = 16;
    int constexpr blocksize = 1;
    using scalarx2_t = sycl::vec<scalar_t, 2>;
    int ld_scale_zp = n;
    if (k % (SgSize * 32 ) == 0) {
      int constexpr TileK = 32;
      int constexpr GroupK = SgSize * TileK;

      int g_idx = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      int g_n = g_idx;
      auto sptr = scales + g_n;
      auto bptr = B + g_n * k;
      auto aptr = A;
      auto cptr = C + g_n;

      sycl::float tmpAcc = 0.f;
      for (int i = 0; i < k; i += GroupK) {
// #pragma unroll
        // for (int iu = 0; iu < Unroll; iu++) {
          uint8_t tmps8[TileK];
          *(sycl::vec<uint8_t, TileK>*)tmps8 =
              *(sycl::vec<uint8_t, TileK>*)(bptr + sg_id * TileK);
          int scale_offset = sg_id * (TileK / blocksize) * ld_scale_zp;
          // int zp_offset = sg_id * (TileK / blocksize) * ld_scale_zp;
          scalar_t scale = *(sptr + scale_offset);
          // scalar_t zero_point = *(zptr + zp_offset);
#pragma unroll
          for (int ikk = 0; ikk < TileK; ikk ++) {
            scalar_t tmpA = *(scalar_t*)(aptr + sg_id * TileK + ikk);
            scalar_t tmpB = static_cast<int8_t>(tmps8[ikk])
            auto tmpAmulB = tmpA * (tmpB * scale + zero_point);
            tmpAcc += tmpAmulB;
          }
          sptr += (GroupK / blocksize) * ld_scale_zp;
          zptr += (GroupK / blocksize) * ld_scale_zp;
          aptr += GroupK;
          bptr += GroupK;
      }
      sycl::float2 sum = {0.f, 0.f};
      sum += sycl::reduce_over_group(sg, tmpAcc, sycl::plus<>());
      if (sg_id == 0) {
        *cptr = static_cast<scalar_t>(sum[0] + sum[1]);
      }
    } else { // k % (SgSize * 32 * Unroll) != 0
      int constexpr TileK = 32;
      int constexpr GroupK = SgSize * TileK;
      int k_body = padto_le(k, GroupK);

      int constexpr TileK2 = 8;
      int constexpr GroupK2 = SgSize * TileK2;
      int k_body2 = padto_le(k, GroupK2);
      int g_idx = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      int g_n = g_idx;
      auto sptr = scales + g_n;
      // auto zptr = ScaleAndZeros + g_n * 2 + 1;
      auto bptr = B + g_n * k;
      auto aptr = A;
      auto cptr = C + g_n;
      sycl::float tmpAcc = 0.f;
      int i = 0;
      for (; i < k_body; i += GroupK) {
// #pragma unroll
        // for (int iu = 0; iu < Unroll; iu++) {
          uint8_t tmps8[TileK];
          *(sycl::vec<uint8_t, TileK>*)tmps8 =
              *(sycl::vec<uint8_t, TileK>*)(bptr + sg_id * TileK);

          int scale_offset = sg_id * TileK / blocksize * ld_scale_zp;
          // int zp_offset = sg_id * TileK / blocksize * ld_scale_zp;

          scalar_t scale = *(sptr + scale_offset);
          // scalar_t zero_point = *(zptr + zp_offset);
#pragma unroll
          for (int ikk = 0; ikk < TileK; ikk ++) {
            scalar_t tmpA = *(scalar_t*)(aptr + sg_id * TileK + ikk);
            scalar_t tmpB = static_cast<int8_t>(tmps8[ikk]);
            auto tmpAmulB = tmpA * (tmpB * scale + zero_point);
            tmpAcc += tmpAmulB;
          }
          sptr += (GroupK / blocksize) * ld_scale_zp;
          zptr += (GroupK / blocksize) * ld_scale_zp;
          aptr += GroupK;
          bptr += GroupK;
        // }
      }
      if (i + GroupK2 < k_body2) {
        for (; i < k_body2; i += GroupK2) {
// #pragma unroll
          // for (int iu = 0; iu < Unroll; iu++) {
            uint8_t tmps8[TileK2];
            *(sycl::vec<uint8_t, TileK2>*)tmps8 =
                *(sycl::vec<uint8_t, TileK2>*)(bptr + sg_id * TileK2);

            int scale_offset = sg_id * TileK2 / blocksize * ld_scale_zp;
            // int zp_offset = sg_id * TileK2 / blocksize * ld_scale_zp;
            scalar_t scale = *(sptr + scale_offset);
            // scalar_t zero_point = *(zptr + zp_offset);
#pragma unroll
            for (int ikk = 0; ikk < TileK2; ikk++) {
              scalar_t tmpA = *(scalar_t*)(aptr + sg_id * TileK2 + ikk);
              scalar_t tmpB = static_cast<int8_t>(tmps8[ikk]);
              auto tmpAmulB = tmpA * (tmpB * scale + zero_point);
              tmpAcc += tmpAmulB;
            }
            sptr += (GroupK2 / blocksize) * ld_scale_zp;
            // zptr += (GroupK2 / blocksize) * ld_scale_zp;
            aptr += GroupK2;
            bptr += GroupK2;
          // }
        }
      }
      if (i + SgSize * 2 <= k) {
        for (; i < k; i += SgSize * 2) {
          uint8_t tmps8 = *(bptr + sg_id);

          int scale_offset = sg_id * 2 / blocksize * ld_scale_zp;
          // int zp_offset = sg_id * 2 / blocksize * ld_scale_zp;
          scalar_t scale = *(sptr + scale_offset);
          // scalar_t zero_point = *(zptr + zp_offset);

          scalar_t tmpB = static_cast<int8_t>(tmps8);
          scalar_t tmpA = *(scalar_t*)(aptr + sg_id * 2);

          auto tmpAmulB = tmpA * (tmpB * scale + zero_point);
          tmpAcc += tmpAmulB;
          sptr += (SgSize * 2 / blocksize) * ld_scale_zp;
          // zptr += (SgSize * 2 / blocksize) * ld_scale_zp;
          aptr += SgSize * 2;
          bptr += SgSize * 2 ;
        }
      }
      sycl::float sum = 0.f;
      sum += sycl::reduce_over_group(sg, tmpAcc, sycl::plus<>());
      if (sg_id == 0) {
        *cptr = static_cast<scalar_t>(sum);
      }
    }
  }

 private:
  const scalar_t* A;
  const uint8_t* B;
  scalar_t* C;
  const scalar_t* scales;
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
};

void linear_int8_kernel(
    const Tensor& A,
    const Tensor& B,
    const Tensor& scales,
    Tensor& C) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  int64_t m = A.size(0);
  int64_t n = C.size(1);
  int64_t k = A.size(1);
  int constexpr SgSize = 16;
  sycl::range<1> local_range{SgSize};
  sycl::range<1> global_range{static_cast<size_t>(n) * SgSize};
  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      A.scalar_type(), "linear_int8_kernel", [&]() {
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

        auto kfn = LinearInt8KernelFunctor<scalar_sycl_t>(
                input_data,
                weight_data,
                output_data,
                scales,
                m,
                n,
                k,
                k,
                k / qGroupSize,
                n);
            sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
            break;
          }
          // case 32: {
          //   auto kfn = LinearInt4KernelFunctor<scalar_sycl_t, 32>(
          //       input_data,
          //       weight_data,
          //       output_data,
          //       scale_zeros_data,
          //       m,
          //       n,
          //       k,
          //       k,
          //       k / qGroupSize,
          //       n);
          //   sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
          //   break;
          // }
          // case 64: {
          //   auto kfn = LinearInt4KernelFunctor<scalar_sycl_t, 64>(
          //       input_data,
          //       weight_data,
          //       output_data,
          //       scale_zeros_data,
          //       m,
          //       n,
          //       k,
          //       k,
          //       k / qGroupSize,
          //       n);
          //   sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
          //   break;
          // }
          // case 128: {
          //   auto kfn = LinearInt4KernelFunctor<scalar_sycl_t, 128>(
          //       input_data,
          //       weight_data,
          //       output_data,
          //       scale_zeros_data,
          //       m,
          //       n,
          //       k,
          //       k,
          //       k / qGroupSize,
          //       n);
          //   sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
          //   break;
          // }
          // case 256: {
          //   auto kfn = LinearInt4KernelFunctor<scalar_sycl_t, 256>(
          //       input_data,
          //       weight_data,
          //       output_data,
          //       scale_zeros_data,
          //       m,
          //       n,
          //       k,
          //       k,
          //       k / qGroupSize,
          //       n);
          //   sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
          //   break;
          // }
        // }
      );
  }

} // namespace at::native::xpu