
#include <ATen/native/xpu/sycl/Dequant_int4.h>
#include <comm/SYCLContext.h>

template <typename scalar_t = sycl::half, int block_size = 32>
struct DequantInt4KernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  DequantInt4KernelFunctor() {}
  void sycl_ker_config_convention(sycl::handler& cgh) {}
  [[intel::reqd_sub_group_size(16)]] void operator()(
      sycl::nd_item<1> it) const {}

 private:
}