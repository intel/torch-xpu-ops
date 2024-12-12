#include <ATen/native/xpu/sycl/LinearInt4.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

template <typename scalar_t = at::Half, int block_size = 16>
struct LinearInt4KernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  LinearInt4KernelFunctor(
      const scalar_t* A,
      const int32_t* B,
      scalar_t* C,
      const scalar_t* B_scale,
      const scalar_t* B_zero_point,
      int m,
      int n,
      int k,
      int lda,
      int ldb,
      int ldc)
      : A(A),
        B(B),
        C(C),
        B_scale(B_scale),
        B_zero_point(B_zero_point),
        m(m),
        n(n),
        k(k),
        lda(lda),
        ldb(ldb),
        ldc(ldc) {}
  void sycl_ker_config_convention(sycl::handler& cgh) {
    // local_scan_ = sycl_local_acc_t<T>(N_, cgh);
  }

  void operator()(sycl::nd_item<1> item) const {
    int constexpr Unroll = 2;
    int constexpr SgSize = 16;
    int constexpr TileK = 32;
    int constexpr GroupK = SgSize * TileK;
    int constexpr blocksize = 16;

    int g_idx = item.get_group(0);
    auto sg = item.get_sub_group();
    int sg_id = sg.get_local_id()[0];
    int g_n = g_idx;
    auto sptr = B_scale + g_n * ldb;
    auto bptr = B + g_n * k / 2;
    auto aptr = A;
    auto cptr = C + g_n;
    if constexpr (std::is_same_v<scalar_t, sycl::half>) {
      sycl::half2 tmpAcc = {0.f, 0.f};
      uint8_t tmps8[TileK / 2];
      for (int i = 0; i < k; i += GroupK * Unroll) {
#pragma unroll
        for (int iu = 0; iu < Unroll; iu++) {
          *(sycl::vec<uint8_t, TileK / 2>*)tmps8 =
              *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
          scalar_t scale = *(sptr + sg_id * TileK / blocksize);
#pragma unroll
          for (int ikk = 0; ikk < TileK; ikk += 2) {
            sycl::half2 tmpA = *(sycl::half2*)&aptr[sg_id * TileK + ikk];
            sycl::half2 tmpB = {
                static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8),
                static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8)};
            tmpAcc += tmpA * tmpB * scale;
          }
          sptr += GroupK / blocksize;
          aptr += GroupK;
          bptr += GroupK / 2;
        }
      }
      sycl::half2 sum = {0.f, 0.f};
      for (int i = 0; i < SgSize; i += 1) {
        sum += group_broadcast(sg, tmpAcc, i);
      }
      if (sg_id == 0) {
        *cptr = sum[0] + sum[1];
      }
    } else {
      scalar_t tmpAcc = 0.f;
      int constexpr Unroll = 2;
      for (int i = 0; i < k; i += GroupK * Unroll) {
#pragma unroll
        for (int iu = 0; iu < Unroll; iu++) {
          uint8_t tmps8[TileK / 2];
          *(sycl::vec<uint8_t, TileK / 2>*)tmps8 =
              *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
          scalar_t scale = *(sptr + sg_id * TileK / blocksize);
#pragma unroll
          for (int ikk = 0; ikk < TileK; ikk += 2) {
            tmpAcc += scalar_t(aptr[sg_id * TileK + ikk]) *
                static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8) * scale;
            tmpAcc += scalar_t(aptr[sg_id * TileK + ikk + 1]) *
                static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8) * scale;
          }
          sptr += GroupK / blocksize;
          aptr += GroupK;
          bptr += GroupK / 2;
        }
      }
      float sum = 0.f;
      for (int i = 0; i < SgSize; i += 1) {
        sum += group_broadcast(sg, tmpAcc, i);
      }
      if (sg_id == 0) {
        *cptr = sum;
      }
    }
  }

 private:
  const scalar_t* A;
  const int32_t* B;
  scalar_t* C;
  const scalar_t* B_scale;
  const scalar_t* B_zero_point;
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
  int64_t n = A.size(1);
  int64_t k = C.size(1);
  int constexpr Unroll = 2;
  int constexpr SgSize = 16;
  sycl::range<1> local_range{SgSize};
  sycl::range<1> global_range{static_cast<size_t>(n) * SgSize};
  int lda = k;
  int ldb = n;
  int ldc = n;

  // AT_DISPATCH_FLOATING_TYPES_AND(
  //     at::ScalarType::Half, A.scalar_type(), "linear_int4_kernel", [&]() {
  if (A.scalar_type() == at::ScalarType::Half) {
    using scalar_t = at::Half;
    const scalar_t* input_data = A.data_ptr<scalar_t>();
    int32_t* weight_data = B.data_ptr<int32_t>(); // int4x8

    scalar_t* output_data = C.data_ptr<scalar_t>();
    scalar_t* weight_scale_data = qScaleAndZeros.data_ptr<scalar_t>();
    LinearInt4KernelFunctor<scalar_t, 16> kfn(
        input_data,
        weight_data,
        output_data,
        weight_scale_data,
        nullptr,
        m,
        n,
        k,
        k,
        n,
        n);

    sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
  }
  // AT_DISPATCH_FLOATING_TYPES_AND(
  //     at::ScalarType::BFloat16, A.scalar_type(), "linear_int4_kernel", [&]()
  //     {
  else if (A.scalar_type() == at::ScalarType::BFloat16) {
    using scalar_t = at::BFloat16;
    const scalar_t* input_data = A.data_ptr<scalar_t>();
    int32_t* weight_data = B.data_ptr<int32_t>(); // int4x8

    scalar_t* output_data = C.data_ptr<scalar_t>();
    scalar_t* weight_scale_data = qScaleAndZeros.data_ptr<scalar_t>();
    LinearInt4KernelFunctor<scalar_t, 16> kfn(
        input_data,
        weight_data,
        output_data,
        weight_scale_data,
        nullptr,
        m,
        n,
        k,
        k,
        n,
        n);

    sycl_kernel_submit(global_range, local_range, sycl_queue, kfn);
  }
}

} // namespace at::native::xpu