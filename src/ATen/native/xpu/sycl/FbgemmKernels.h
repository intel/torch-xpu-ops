#pragma once

#include <ATen/ATen.h>

#include <sycl/sycl.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace at {
namespace native {
namespace xpu {

constexpr size_t kStackArrayMaxDims = 5;

template <typename T, typename V>
inline auto CeilDiv(T a, V b) {
  return (a + b - 1) / b;
}

void dense_to_jagged_forward_xpu_kernel(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values);

void jagged_to_padded_dense_forward_xpu_kernel(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    const double padding_value = 0.0);

// For SYCL free function
template <auto* kptr, int RANGE_DIM, typename... Kargs>
inline void sycl_kernel_submit(
    ::sycl::range<RANGE_DIM> global_range,
    ::sycl::range<RANGE_DIM> local_range,
    ::sycl::queue q,
    Kargs... args) {
  sycl::context ctxt = q.get_context();
  auto exe_bndl =
      syclexp::get_kernel_bundle<kptr, sycl::bundle_state::executable>(ctxt);
  sycl::kernel ker = exe_bndl.template ext_oneapi_get_kernel<kptr>();
  syclexp::launch_config cfg{
      ::sycl::nd_range<RANGE_DIM>(global_range, local_range)};
  syclexp::nd_launch(q, cfg, ker, args...);
}

} // namespace xpu
} // namespace native
} // namespace at
