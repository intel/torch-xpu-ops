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
      int ldc,
      sycl::stream& os)
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

  void operator()(sycl::nd_item<1> it) const {
    int constexpr Unroll = 2;
    int constexpr SgSize = 16;
    int constexpr blocksize = block_size;

    if (k % (SgSize * 32 * Unroll) == 0) {
      int constexpr TileK = 32;
      int constexpr GroupK = SgSize * TileK;

      int g_idx = it.get_group(0);
      auto sg = it.get_sub_group();
      int sg_id = sg.get_local_id()[0];
      int g_n = g_idx;
      auto sptr = ScaleAndZeros + g_n * ldb * 2;
      auto bptr = B + g_n * k / 2;
      auto aptr = A;
      auto cptr = C + g_n;
      if constexpr (std::is_same_v<scalar_t, sycl::half>) { // Half
        sycl::half2 tmpAcc = {0.f, 0.f};
        for (int i = 0; i < k; i += GroupK * Unroll) {
#pragma unroll
          for (int iu = 0; iu < Unroll; iu++) {
            uint8_t tmps8[TileK / 2];
            *(sycl::vec<uint8_t, TileK / 2>*)tmps8 =
                *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
            scalar_t scale = *(sptr + sg_id * (TileK / blocksize) * 2);
            scalar_t zero_point = *(sptr + sg_id * (TileK / blocksize) * 2 + 1);
#pragma unroll
            for (int ikk = 0; ikk < TileK; ikk += 2) {
              sycl::half2 tmpA = *(sycl::half2*)&aptr[sg_id * TileK + ikk];
              sycl::half2 tmpB = {
                  static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8),
                  static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8)};

              tmpAcc += tmpA * (tmpB * scale + zero_point);
            }
            sptr += (GroupK / blocksize) * 2;
            aptr += GroupK;
            bptr += GroupK / 2;
          }
        }
        sycl::half2 sum = {0.f, 0.f};
        for (int i = 0; i < SgSize; i += 1) {
          sum += select_from_group(sg, tmpAcc, i);
        }
        if (sg_id == 0) {
          *cptr = sum[0] + sum[1];
        }
      } else { // Bfloat16
        scalar_t tmpAcc = 0.f;
        int constexpr Unroll = 2;
        for (int i = 0; i < k; i += GroupK * Unroll) {
#pragma unroll
          for (int iu = 0; iu < Unroll; iu++) {
            uint8_t tmps8[TileK / 2];
            *(sycl::vec<uint8_t, TileK / 2>*)tmps8 =
                *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
            scalar_t scale = *(sptr + (sg_id * TileK / blocksize) * 2);
            scalar_t zero_point = *(sptr + (sg_id * TileK / blocksize) * 2 + 1);
#pragma unroll
            for (int ikk = 0; ikk < TileK; ikk += 2) {
              tmpAcc += scalar_t(aptr[sg_id * TileK + ikk]) *
                  (static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8) * scale +
                   zero_point);
              tmpAcc += scalar_t(aptr[sg_id * TileK + ikk + 1]) *
                  (static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8) * scale +
                   zero_point);
            }
            sptr += (GroupK / blocksize) * 2;
            aptr += GroupK;
            bptr += GroupK / 2;
          }
        }
        float sum = 0.f;
        for (int i = 0; i < SgSize; i += 1) {
          sum += select_from_group(sg, tmpAcc, i);
        }
        if (sg_id == 0) {
          *cptr = sum;
        }
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
      auto bptr = B + g_n * k / 2;
      auto aptr = A;
      auto cptr = C + g_n;
      if constexpr (std::is_same_v<scalar_t, sycl::half>) { // Half
        sycl::half2 tmpAcc = {0.f, 0.f};
        sycl::vec<float, 2> tmpAcc_fp32{0.f, 0.f};
        int i = 0;
        for (; i < k_body; i += GroupK * Unroll) {
#pragma unroll
          for (int iu = 0; iu < Unroll; iu++) {
            uint8_t tmps8[TileK / 2];
            *(sycl::vec<uint8_t, TileK / 2>*)tmps8 =
                *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);

            // scalar_t scale = *(sptr + sg_id * TileK / blocksize);
            scalar_t scale = *(sptr + sg_id * (TileK / blocksize) * 2);
            scalar_t zero_point = *(sptr + sg_id * (TileK / blocksize) * 2 + 1);
#pragma unroll
            for (int ikk = 0; ikk < TileK; ikk += 2) {
              sycl::half2 tmpA = *(sycl::half2*)&aptr[sg_id * TileK + ikk];
              sycl::half2 tmpB = {
                  static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8),
                  static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8)};
              // tmpAcc += tmpA * tmpB * scale;
              tmpAcc += tmpA * (tmpB * scale + zero_point);
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
              // scalar_t scale = *(sptr + sg_id * TileK2 / blocksize);
              scalar_t scale = *(sptr + sg_id * (TileK2 / blocksize) * 2);
              scalar_t zero_point =
                  *(sptr + sg_id * (TileK2 / blocksize) * 2 + 1);
#pragma unroll
              for (int ikk = 0; ikk < TileK2; ikk += 2) {
                sycl::half2 tmpA = *(sycl::half2*)&aptr[sg_id * TileK2 + ikk];
                sycl::half2 tmpB = {
                    static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8),
                    static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8)};
                tmpAcc += tmpA * (tmpB * scale + zero_point);
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
            sycl::half2 tmpB = {
                static_cast<int8_t>((tmps8 & 0x0f) - 8),
                static_cast<int8_t>((tmps8 >> 4) - 8)};
            scalar_t scale = *(sptr + (sg_id * 2 / blocksize) * 2);
            scalar_t zero_point = *(sptr + (sg_id * 2 / blocksize) * 2 + 1);
            sycl::half2 tmpA = *(sycl::half2*)&aptr[sg_id * 2];
            tmpAcc += tmpA * (tmpB * scale + zero_point);
            sptr += (SgSize * 2 / blocksize) * 2;
            aptr += SgSize * 2;
            bptr += SgSize * 2 / 2;
          }
        }
        sycl::half2 sum = {0.f, 0.f};
        sum += sycl::reduce_over_group(sg, tmpAcc, sycl::plus<sycl::half2>());
        if (sg_id == 0) {
          *cptr = sum[0] + sum[1];
        }
      } else { // Bfloat16
        scalar_t tmpAcc = 0.f;
        int constexpr Unroll = 2;
        int i = 0;
        for (; i < k_body; i += GroupK * Unroll) {
#pragma unroll
          for (int iu = 0; iu < Unroll; iu++) {
            uint8_t tmps8[TileK / 2];
            *(sycl::vec<uint8_t, TileK / 2>*)tmps8 =
                *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
            scalar_t scale = *(sptr + (sg_id * TileK / blocksize) * 2);
            scalar_t zero_point = *(sptr + (sg_id * TileK / blocksize) * 2 + 1);
#pragma unroll
            for (int ikk = 0; ikk < TileK; ikk += 2) {
              tmpAcc += scalar_t(aptr[sg_id * TileK + ikk]) *
                  (static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8) * scale +
                   zero_point);
              tmpAcc += scalar_t(aptr[sg_id * TileK + ikk + 1]) *
                  (static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8) * scale +
                   zero_point);
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
              scalar_t scale = *(sptr + (sg_id * TileK2 / blocksize) * 2);
              scalar_t zero_point =
                  *(sptr + (sg_id * TileK2 / blocksize) * 2 + 1);
#pragma unroll
              for (int ikk = 0; ikk < TileK2; ikk += 2) {
                tmpAcc += scalar_t(aptr[sg_id * TileK2 + ikk]) *
                    (static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8) * scale +
                     zero_point);
                tmpAcc += scalar_t(aptr[sg_id * TileK2 + ikk + 1]) *
                    (static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8) * scale +
                     zero_point);
              }
              sptr += (GroupK2 / blocksize) * 2;
              aptr += GroupK2;
              bptr += GroupK2 / 2;
            }
          }
        }
        if (i + SgSize * Unroll <= k) {
          for (; i < k; i += 2 * SgSize) {
            uint8_t tmps8 = *(bptr + sg_id);
            scalar_t scale = *(sptr + (sg_id * 2 / blocksize) * 2);
            scalar_t zero_point = *(sptr + (sg_id * 2 / blocksize) * 2 + 1);
            tmpAcc += scalar_t(aptr[sg_id * 2]) *
                (static_cast<int8_t>((tmps8 & 0x0f) - 8) * scale + zero_point);
            tmpAcc += scalar_t(aptr[sg_id * 2 + 1]) *
                (static_cast<int8_t>((tmps8 >> 4) - 8) * scale + zero_point);
            sptr += (SgSize * 2 / blocksize) * 2;
            aptr += SgSize * 2;
            bptr += SgSize * 2 / 2;
          }
        }
        float sum = 0.f;
        for (int i = 0; i < SgSize; i += 1) {
          sum += select_from_group(sg, tmpAcc, i);
        }
        // sum += sycl::reduce_over_group(sg, tmpAcc, sycl::plus<scalar_t>());
        if (sg_id == 0) {
          *cptr = sum;
        }
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
  int constexpr Unroll = 2;
  int constexpr SgSize = 16;
  sycl::range<1> local_range{SgSize};
  sycl::range<1> global_range{static_cast<size_t>(n) * SgSize};

  if (A.scalar_type() == at::ScalarType::Half) {
    using scalar_t = at::Half;
    using scalar_sycl_t = sycl::half;
    const scalar_sycl_t* input_data =
        reinterpret_cast<scalar_sycl_t*>(A.data_ptr<scalar_t>());
    uint8_t* weight_data =
        reinterpret_cast<uint8_t*>(B.data_ptr<uint8_t>()); // int4x2

    scalar_sycl_t* output_data =
        reinterpret_cast<scalar_sycl_t*>(C.data_ptr<scalar_t>());
    scalar_sycl_t* scale_zeros_data =
        reinterpret_cast<scalar_sycl_t*>(qScaleAndZeros.data_ptr<scalar_t>());
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

    // auto cgf = [&](::sycl::handler& cgh) {
    //   auto os = sycl::stream(1024 * 8, 768 * 8, cgh);
    //   LinearInt4KernelFunctor<scalar_sycl_t, 32> kfn(
    //       input_data,
    //       weight_data,
    //       output_data,
    //       scale_zeros_data,
    //       m,
    //       n,
    //       k,
    //       k,
    //       k / qGroupSize,
    //       n,
    //       os);
    //   kfn.sycl_ker_config_convention(cgh);
    //   cgh.parallel_for<LinearInt4KernelFunctor<scalar_sycl_t, 32>>(
    //       ::sycl::nd_range<1>(global_range, local_range), kfn);
    // };
    // sycl_queue.submit(cgf);

  }
  // AT_DISPATCH_FLOATING_TYPES_AND(
  //     at::ScalarType::BFloat16, A.scalar_type(), "linear_int4_kernel",
  //     [&]()
  //     {
  else if (A.scalar_type() == at::ScalarType::BFloat16) {
    using scalar_t = at::BFloat16;
    using scalar_sycl_t = sycl::ext::oneapi::bfloat16;
    const scalar_sycl_t* input_data =
        reinterpret_cast<scalar_sycl_t*>(A.data_ptr<scalar_t>());
    uint8_t* weight_data =
        reinterpret_cast<uint8_t*>(B.data_ptr<uint8_t>()); // int4x2

    scalar_sycl_t* output_data =
        reinterpret_cast<scalar_sycl_t*>(C.data_ptr<scalar_t>());
    scalar_sycl_t* scale_zeros_data =
        reinterpret_cast<scalar_sycl_t*>(qScaleAndZeros.data_ptr<scalar_t>());
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

    // auto cgf = [&](::sycl::handler& cgh) {
    //   auto os = sycl::stream(1024 * 8, 768 * 8, cgh);
    //   LinearInt4KernelFunctor<scalar_sycl_t, 32> kfn(
    //       input_data,
    //       weight_data,
    //       output_data,
    //       scale_zeros_data,
    //       m,
    //       n,
    //       k,
    //       k,
    //       k / qGroupSize,
    //       n,
    //       os);
    //   kfn.sycl_ker_config_convention(cgh);
    //   cgh.parallel_for<LinearInt4KernelFunctor<scalar_sycl_t, 32>>(
    //       ::sycl::nd_range<1>(global_range, local_range), kfn);
    // };
    // sycl_queue.submit(cgf);
  }
}

} // namespace at::native::xpu