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

uint32_t xpu_calc_xblock_count_base(int num_items, int threads_per_block) {
  // The number of threads can be as high as 2048 on some newer architectures,
  // but this is not portable.
  TORCH_CHECK(
      threads_per_block <= syclDeviceMaxWorkGroupSize(),
      "Number of threads must be <=1024!");
  constexpr uint64_t max_blocks = 2147483647;
  const auto u_num_items = static_cast<uint64_t>(num_items);
  const auto u_threads = static_cast<uint64_t>(threads_per_block);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks =
      u_num_items / u_threads + (u_num_items % u_threads != 0);
  return static_cast<uint32_t>(std::min(blocks, max_blocks));
}

// See: xpu_calc_xblock_count_base
uint32_t xpu_calc_xblock_count(int num_items, int threads_per_block) {
  TORCH_CHECK(
      num_items >= 0,
      "When calculating block counts, the number of items must be positive!");
  return xpu_calc_xblock_count_base(num_items, threads_per_block);
}

constexpr size_t kStackArrayMaxDims = 5;

template <typename T, typename V>
inline auto CeilDivUp(T a, V b) {
  return (a + b - 1) / b;
}

template <typename T, typename V>
inline auto round_down(T a, V b) {
  return a / b * b;
}

inline bool torch_tensor_undefined(const at::Tensor& ten) {
  return ten.defined();
}

inline bool torch_tensor_undefined(const std::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_undefined(ten.value());
}

inline bool torch_tensor_on_xpu_check(const at::Tensor& ten) {
  return ten.is_xpu();
}

inline bool torch_tensor_on_xpu_check(const std::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_on_xpu_check(ten.value());
}

inline std::optional<int64_t> get_device_index_from_tensor(
    const at::Tensor& ten) {
  return {ten.device().index()};
}

inline std::optional<int64_t> get_device_index_from_tensor(
    const std::optional<at::Tensor>& ten) {
  if (ten) {
    return {ten->device().index()};
  } else {
    return {};
  }
}

inline std::string torch_tensor_device_name(const at::Tensor& ten) {
  return c10::DeviceTypeName(ten.device().type());
}

inline std::string torch_tensor_device_name(
    const std::optional<at::Tensor>& ten) {
  if (ten.has_value()) {
    return torch_tensor_device_name(ten.value());
  } else {
    return "N/A";
  }
}

template <typename... Tensors>
std::string tensor_on_same_xpu_if_not_optional_check(
    const std::string& var_names_str,
    const Tensors&... tensors) {
  std::optional<int64_t> xpu_index;
  bool on_same_xpu = true;

  // Collect the XPU index of the first non-empty optional tensor and make sure
  // that all tensors are on this same index.
  (
      [&](const auto& tensor) {
        if (!torch_tensor_undefined(tensor)) {
          return;
        }
        if (!torch_tensor_on_xpu_check(tensor)) {
          on_same_xpu = false;
          return;
        }
        const auto my_xpu_index = get_device_index_from_tensor(tensor);
        if (my_xpu_index) {
          if (!xpu_index) {
            xpu_index = my_xpu_index;
          } else if (*xpu_index != my_xpu_index) {
            on_same_xpu = false;
          }
        }
      }(tensors),
      ...);

  if (on_same_xpu) {
    return "";
  }

  std::vector<std::string> var_names;
  {
    std::string temp;
    for (const auto& x : var_names_str) {
      if (x == ',') {
        var_names.push_back(temp);
        temp = "";
      } else {
        temp.push_back(x);
      }
    }
    var_names.push_back(temp);
  }

  // Not all the tensors on a GPU or on the same GPU, generate a message.
  std::string msg = "Not all tensors were on the same GPU: ";
  size_t current_idx = 0;
  (
      [&](const auto& tensor) {
        if (current_idx > 0) {
          msg.append(", ");
        }
        msg.append(
            var_names.at(current_idx++) + "(" +
            torch_tensor_device_name(tensor));
        const auto xpu_device_index = get_device_index_from_tensor(tensor);
        if (xpu_device_index) {
          msg.append(":" + std::to_string(*xpu_device_index));
        }
        msg.append(")");
      }(tensors),
      ...);

  return msg;
}

#define TENSORS_ON_SAME_XPU_IF_NOT_OPTIONAL(...)                             \
  do {                                                                       \
    const auto tensors_on_same_xpu =                                         \
        tensor_on_same_xpu_if_not_optional_check(#__VA_ARGS__, __VA_ARGS__); \
    TORCH_CHECK(tensors_on_same_xpu.empty(), tensors_on_same_xpu);           \
  } while (false)

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

void reorder_batched_ad_lengths_xpu_kernel(
    const Tensor& cat_ad_lengths,
    const Tensor& batch_offsets,
    Tensor& reordered_cat_ad_lengths,
    const int32_t T,
    const bool broadcast_lengths,
    const int32_t grid_size);

void reorder_batched_ad_indices_xpu_kernel(
    const at::Tensor& cat_ad_offsets,
    const at::Tensor& cat_ad_indices,
    const at::Tensor& reordered_cat_ad_offsets,
    const at::Tensor& batch_offsets,
    at::Tensor& reordered_cat_ad_indices,
    const int64_t num_ads_in_batch,
    const int64_t B,
    const int64_t T,
    const bool broadcast_indices = false);

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

