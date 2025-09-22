#pragma once

#include <ATen/ATen.h>

#include <sycl/sycl.hpp>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace at {

template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

namespace native {
namespace xpu {

constexpr size_t kStackArrayMaxDims = 5;

template <typename T, typename V>
inline auto CeilDivUp(T a, V b) {
  return (a + b - 1) / b;
}

template <typename T, typename V>
inline auto round_down(T a, V b) {
  return a / b * b;
}

struct VecType128 {
  typedef sycl::float4 TType; // Transaction Type
  typedef struct __attribute__((aligned(16))) {
    sycl::half a, b, c, d, w, x, y, z;
  } half8;

  union Data {
    half8 val;
    TType mask;
    Data() {
      mask = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
  } data;

  // VecType128() {
  // data.mask = sycl::float4(0.0f, 0.0f, 0.0f, 0.0f);
  // }
};

struct VecType64 {
  typedef sycl::vec<float, 2> TType; // Transaction Type
  typedef struct __attribute__((aligned(8))) {
    sycl::half a, b, c, d;
  } half4;

  union Data {
    half4 val;
    TType mask;
    Data() {
      mask = sycl::vec<float, 2>(0.0f, 0.0f);
    }
  } data;

  // VecType64() {
  // data.mask = sycl::vec<float, 2>(0.0f, 0.0f);
  // }
};

struct VecType32 {
  typedef float TType; // Transaction Type

  union Data {
    sycl::vec<sycl::half, 2> val;
    TType mask;
    Data() {
      mask = 0.0f;
    }
  } data;

  // VecType32() {
  //   data.mask = 0.0f;
  // }
};

template <typename F>
void f128(
    VecType128& v_out,
    const VecType128& x,
    const VecType128& y0,
    const VecType128& y1,
    F f) {
  v_out.data.val.a = f(x.data.val.a, y0.data.val.a, y1.data.val.a);
  v_out.data.val.b = f(x.data.val.b, y0.data.val.b, y1.data.val.b);
  v_out.data.val.c = f(x.data.val.c, y0.data.val.c, y1.data.val.c);
  v_out.data.val.d = f(x.data.val.d, y0.data.val.d, y1.data.val.d);
  v_out.data.val.w = f(x.data.val.w, y0.data.val.w, y1.data.val.w);
  v_out.data.val.x = f(x.data.val.x, y0.data.val.x, y1.data.val.x);
  v_out.data.val.y = f(x.data.val.y, y0.data.val.y, y1.data.val.y);
  v_out.data.val.z = f(x.data.val.z, y0.data.val.z, y1.data.val.z);
}

template <typename F>
void f64(
    VecType64& v_out,
    const VecType64& x,
    const VecType64& y0,
    const VecType64& y1,
    F f) {
  v_out.data.val.a = f(x.data.val.a, y0.data.val.a, y1.data.val.a);
  v_out.data.val.b = f(x.data.val.b, y0.data.val.b, y1.data.val.b);
  v_out.data.val.c = f(x.data.val.c, y0.data.val.c, y1.data.val.c);
  v_out.data.val.d = f(x.data.val.d, y0.data.val.d, y1.data.val.d);
}

template <typename F>
void f32(
    VecType32& v_out,
    const VecType32& x,
    const VecType32& y0,
    const VecType32& y1,
    F f) {
  v_out.data.val = sycl::vec<sycl::half, 2>(
      f(x.data.val.x(), y0.data.val.x(), y1.data.val.x()),
      f(x.data.val.y(), y0.data.val.y(), y1.data.val.y()));
}

template <typename F>
void fh(
    sycl::half& v_out,
    const sycl::half& x,
    const sycl::half& y0,
    const sycl::half& y1,
    F f) {
  v_out = f(x, y0, y1);
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

void jagged_dense_elementwise_add_jagged_output_fwd_xpu_kn(
    const Tensor& x_values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense,
    const Tensor& output_values);

// For SYCL free function
template <auto* kptr, int RANGE_DIM, typename... Kargs>
inline void sycl_kernel_submit(
    ::sycl::range<RANGE_DIM> global_range,
    ::sycl::range<RANGE_DIM> local_range,
    ::sycl::queue q,
    uint32_t slm_sz,
    Kargs... args) {
  sycl::context ctxt = q.get_context();
  auto exe_bndl =
      syclexp::get_kernel_bundle<kptr, sycl::bundle_state::executable>(ctxt);
  sycl::kernel ker = exe_bndl.template ext_oneapi_get_kernel<kptr>();
  if (slm_sz == 0) {
    syclexp::launch_config cfg{
        ::sycl::nd_range<RANGE_DIM>(global_range, local_range)};
    syclexp::nd_launch(q, cfg, ker, args...);
  } else {
    syclexp::launch_config cfg{
        ::sycl::nd_range<RANGE_DIM>(global_range, local_range),
        syclexp::properties{syclexp::work_group_scratch_size(slm_sz)}};
    syclexp::nd_launch(q, cfg, ker, args...);
  }
}

} // namespace xpu
} // namespace native
} // namespace at
