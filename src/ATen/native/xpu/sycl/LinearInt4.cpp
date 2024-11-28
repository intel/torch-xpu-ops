#include <ATen/native/xpu/sycl/LinearInt4.h>

namespace at::native::xpu {

void linear_int4_kernel(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& weight_scale_zero_point,
    const std::optional<Tensor>& weight_bias,
    Tensor& output,
    int block_size) {
  auto& sycl_queue = at::xpu::getCurrentSYCLQueue();
  int64_t m = input[0];
  int64_t n = input[1];
  int64_t k = output[1];
  int constexpr Unroll = 2;
  int constexpr SgSize = 16;
  sycl::range<1> group{SgSize};
  sycl::range<1> problem{static_cast<size_t>(n) * SgSize};
  int lda = k;
  int ldb = n;
  int ldc = n;
  scalar_t* input_data = input.data_ptr<scalar_t>();
  int4x8* weight_data = weight.data_ptr<int4x8>();
  scalar_t* output_data = output.data_ptr<scalar_t>();
  scalar_t* weight_scale_data = weight_scale.data_ptr<scalar_t>();
  auto kfn = LinearInt4KernelFunctor<sycl::half, 16>(
      input_data,
      weight_data,
      output_data,
      weight_scale_data,
      nullptr,
      m,
      n,
      k,
      n,
      n);
  sycl_kernel_submit(::sycl::nd_range<1>(problem, group), sycl_queue, kfn);
}

template <typename scalar_t, int block_size>
struct LinearInt4KernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  LinearInt4KernelFunctor(
      scalar_t* A,
      int4x8* B,
      scalar_t* C,
      scalar_t* B_scale,
      scalar_t* B_zero_point,
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
      for (int i = 0; i < k; i += GroupK * Unroll) {
#pragma unroll
        for (int iu = 0; iu < Unroll; iu++) {
          uint8_t tmps8[TileK / 2];
          *(sycl::vec<uint8_t, TileK / 2>*)tmps8 =
              *(sycl::vec<uint8_t, TileK / 2>*)(bptr + sg_id * TileK / 2);
          scale_t scale = *(sptr + sg_id * TileK / blocksize);
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
        sum += sg.shuffle(tmpAcc, i);
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
          scale_t scale = *(sptr + sg_id * TileK / blocksize);
#pragma unroll
          for (int ikk = 0; ikk < TileK; ikk += 2) {
            tmpAcc += scale_t(aptr[sg_id * TileK + ikk]) *
                static_cast<int8_t>((tmps8[ikk / 2] & 0x0f) - 8) * scale;
            tmpAcc += scale_t(aptr[sg_id * TileK + ikk + 1]) *
                static_cast<int8_t>((tmps8[ikk / 2] >> 4) - 8) * scale;
          }
          sptr += GroupK / blocksize;
          aptr += GroupK;
          bptr += GroupK / 2;
        }
      }
      float sum = 0.f;
      for (int i = 0; i < SgSize; i += 1) {
        sum += sg.shuffle(tmpAcc, i);
      }
      if (sg_id == 0) {
        *cptr = sum;
      }
    }
  }

 private:
  scalar_t* A;
  int4x8* B;
  scalar_t* C;
  scalar_t* B_scale;
  scalar_t* B_zero_point;
  int m;
  int n;
  int k;
  int lda;
  int ldb;
  int ldc;
};
} // namespace at::native::xpu