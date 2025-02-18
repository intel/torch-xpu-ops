#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <comm/SYCLContext.h>

#include <ATen/native/xpu/sycl/WeightInt4PackKernel.h>

namespace at::native::xpu {

struct WeightToInt4PackKernelFunctor {
  void operator()(sycl::item<1> item) const {
    auto idx = item.get_linear_id();
    int K_div_2 = K_ / 2;
    int K_div_8 = K_ / 8;
    int out_y = idx / K_div_8;
    int out_x = idx % K_div_8;
    int in_y = out_y;
    int in_x = out_x * 4;

    using vec_t = memory::aligned_vector<uint8_t, 4>;
    vec_t input = *reinterpret_cast<vec_t*>(&weight_[in_y * K_div_2 + in_x]);
    vec_t output;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      output[i] = input[i];
    }
    *reinterpret_cast<vec_t*>(&weight_packed_[out_y * K_div_8 + out_x]) = output;
  }
  WeightToInt4PackKernelFunctor(
      uint32_t* weight_packed,
      uint8_t* weight,
      int N,
      int K)
      : weight_packed_(weight_packed), weight_(weight), N_(N), K_(K) {}

 private:
  uint32_t* weight_packed_;
  uint8_t* weight_;
  int N_;
  int K_;
};

void weight_to_int4pack_kernel(
    const Tensor& weight_packed,
    const Tensor& weight,
    int N,
    int K) {
  auto weight_packed_data =
      reinterpret_cast<uint32_t*>(weight_packed.data_ptr());
  const auto weight_data = weight.data_ptr<uint8_t>();
  int K_div_8 = K / 8;
  size_t global_range = N * K_div_8;
  auto fn =
      WeightToInt4PackKernelFunctor(weight_packed_data, weight_data, N, K);
  sycl_kernel_submit(sycl::range<1>(global_range), getCurrentSYCLQueue(), fn);
}

} // namespace at::native::xpu
