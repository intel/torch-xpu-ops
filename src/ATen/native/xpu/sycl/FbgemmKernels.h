#pragma once

#include <ATen/ATen.h>

#include <sycl/sycl.hpp>

#include <comm/SYCLContext.h>

namespace syclext = sycl::ext::oneapi;
namespace syclexp = sycl::ext::oneapi::experimental;

namespace at {

template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

namespace native {
namespace xpu {

#define FBGEMM_DISPATCH_FLOAT_AND_BFLOAT16_CASE(...)   \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define FBGEMM_DISPATCH_FLOATING_TYPES_CASE(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define FBGEMM_DISPATCH_ALL_TYPES(TYPE, NAME, ...)     \
  AT_DISPATCH_SWITCH(                                  \
      TYPE,                                            \
      NAME,                                            \
      FBGEMM_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__) \
          FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__))

#define FBGEMM_DISPATCH_ALL_TYPES_AND_DOUBLE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                         \
      TYPE,                                                   \
      NAME,                                                   \
      FBGEMM_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__)        \
          FBGEMM_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__)    \
              AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__))

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

TORCH_XPU_API void dense_to_jagged_forward_xpu_kernel(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output_values);

TORCH_XPU_API void jagged_to_padded_dense_forward_xpu_kernel(
    const Tensor& x_values,
    const std::vector<Tensor>& x_offsets,
    const Tensor& y,
    const Tensor& output,
    const double padding_value = 0.0);

TORCH_XPU_API void jagged_dense_elementwise_add_jagged_output_fwd_xpu_kn(
    const Tensor& x_values,
    const std::vector<Tensor>& offsets,
    const Tensor& dense,
    const Tensor& output_values);

TORCH_XPU_API void reorder_batched_ad_lengths_xpu_kernel(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    Tensor& reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths,
    const int32_t grid_size);

TORCH_XPU_API void fbgemm_cumsum_kernel(
    const Tensor& result,
    const Tensor& self,
    int64_t dim);

TORCH_XPU_API void reorder_batched_ad_indices_xpu_kernel(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    at::Tensor& reordered_cat_ad_indices,
    const int64_t num_ads_in_batch,
    const int64_t B,
    const int64_t T,
    const bool broadcast_indices = false);

TORCH_XPU_API void permute_2D_lengths_kernel_xpu(
    int32_t T,
    int32_t B,
    const at::Tensor& lengths_contig,
    const at::Tensor& permute_contig,
    at::Tensor& permuted_lengths);

TORCH_XPU_API void permute_2D_data_kernel_xpu(
    int32_t permuted_indices_size,
    int32_t T,
    int32_t B,
    const Tensor& indices_contig,
    const std::optional<const Tensor>& weights,
    const int32_t weights_columns,
    const Tensor& permute_contig,
    const Tensor& input_offsets,
    const Tensor& output_offsets,
    Tensor& permuted_indices,
    const std::optional<Tensor>& permuted_weights);

} // namespace xpu
} // namespace native
} // namespace at
