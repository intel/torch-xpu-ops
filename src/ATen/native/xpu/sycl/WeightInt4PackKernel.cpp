#include <ATen/ATen.h>
#include <ATen/native/xpu/sycl/MemoryAccessUtils.h>
#include <ATen/native/xpu/sycl/MaxUnpoolingKernels.h>
#include <comm/SYCLContext.h>

namespace at::native::xpu {

#define SIMD32 32
#define SIMD16 16

template <typename T, typename V>
inline auto CeilDiv(T a, V b) {
  return (a + b - 1) / b;
}

template <typename vec_t, typename in_t, typename out_t, int vec_len>
struct WeightToInt4packKernelFunctor {
  WeightToInt4packKernelFunctor(
      out_t* res_ptr_,
      in_t* in_ptr_,
      const int N_,
      const int K_,
      const int fold_len_,
      const uint64_t stride_k_)
      : res_ptr(res_ptr_),
        in_ptr(in_ptr_),
        N(N_),
        K(K_),
        fold_len(fold_len_),
        stride_k(stride_k_) {}
  void operator()(sycl::nd_item<2> item) const {
    uint64_t g_row = item.get_global_id()[0];
    uint64_t g_loc = item.get_global_id()[1];
    uint64_t k_index = g_loc * vec_len;
    if ((g_row < N) && (k_index < K)) {
      vec_t even =
          *(reinterpret_cast<vec_t*>(in_ptr + g_row * fold_len * K + k_index));
      vec_t odd = *(reinterpret_cast<vec_t*>(
          in_ptr + g_row * fold_len * K + K + k_index));
#pragma unroll
      for (int i = 0; i < vec_len; i++) {
        res_ptr[(k_index + i) * N + g_row] =
            (((out_t)(odd[i])) << 4) | ((out_t)(even[i]));
      }
    }
  }

 private:
  out_t* res_ptr;
  in_t* in_ptr;
  int N;
  int K;
  int fold_len;
  uint64_t stride_k;
};

template <typename in_t, typename out_t, int vec_len>
struct WeightToInt4packKernelFunctor<int32_t, in_t, out_t, vec_len> {
  WeightToInt4packKernelFunctor(
      out_t* res_ptr_,
      in_t* in_ptr_,
      const int N_,
      const int K_,
      const int fold_len_,
      const uint64_t stride_k_)
      : res_ptr(res_ptr_),
        in_ptr(in_ptr_),
        N(N_),
        K(K_),
        fold_len(fold_len_),
        stride_k(stride_k_) {}
  void operator()(sycl::nd_item<2> item) const {
    uint64_t g_row = item.get_global_id()[0];
    uint64_t g_loc = item.get_global_id()[1];
#pragma unroll
    for (int i = 0; i < vec_len; i++) {
      uint64_t k_index = g_loc + i * stride_k;
      if ((g_row < N) && (k_index < K)) {
        out_t even = (out_t)(in_ptr[g_row * fold_len * K + k_index]);
        out_t odd = (out_t)(in_ptr[g_row * fold_len * K + K + k_index]);
        res_ptr[k_index * N + g_row] = ((odd << 4) | even);
      }
    }
  }

 private:
  out_t* res_ptr;
  in_t* in_ptr;
  int N;
  int K;
  int fold_len;
  uint64_t stride_k;
};

void get_float_len(
  int K, 
  int SIMD, 
  uint64_t& float_len) {
  if ((SIMD == SIMD32) && (K <= 16)) {
    float_len = 1;
  } else if (K > 16 && K <= 32) {
    float_len = 1;
  } else if (K > 32 && K <= 64) {
    float_len = 2;
  } else {
    float_len = 4;
  }
}

void get_group_param(
    int N,
    int K,
    int fold_len,
    uint64_t& global_row,
    uint64_t& global_col,
    uint64_t& local_row,
    uint64_t& local_col,
    uint64_t& float_len,
    uint64_t& stride_k,
    int64_t maxWGSize,
    int SIMD) {
  if ((SIMD == SIMD32) && (K <= 16)) {
    SIMD = SIMD16;
    local_col = SIMD;
    local_row = maxWGSize / local_col;
    global_col = 1;
    global_row = CeilDiv(static_cast<uint64_t>(N / fold_len), local_row);
    stride_k = global_col * local_col;
    return;
  }
  local_col = SIMD;
  local_row = maxWGSize / local_col;
  global_col = CeilDiv(static_cast<uint64_t>(K), local_col * float_len);
  global_row = CeilDiv(static_cast<uint64_t>(N / fold_len), local_row);
  stride_k = global_col * local_col;
}

void weight_to_int4pack_kernel(
    const Tensor& weight_packed,
    const Tensor& weight,
    int N,
    int K,
    int fold_len) {
  using in_t = int32_t;
  using out_t = uint8_t;
  // int fold_len = 2;
  uint64_t global_row = 0, global_col = 0, local_row = 0, local_col = 0,
           float_len = 0, stride_k = 0;
  auto* dev_prop =
      at::xpu::getDeviceProperties(at::xpu::getDeviceIndexOfCurrentQueue());
  auto sub_group_size = dev_prop->sub_group_sizes;
  int SIMD = sub_group_size[1];
  get_float_len(K, SIMD, float_len);
  auto weight_packed_data = reinterpret_cast<out_t*>(weight_packed.data_ptr());
  const auto weight_data = weight.data_ptr<in_t>();

#define VEC_WEIGHT_PACK_KERNEL_FUNC(vec_len, vec_t)                            \
  {                                                                            \
    WeightToInt4packKernelFunctor<vec_t, in_t, out_t, vec_len> kfn(            \
        weight_packed_data, weight_data, N / fold_len, K, fold_len, stride_k); \
    int64_t maxWGSize = syclMaxWorkGroupSize(kfn);                             \
    get_group_param(                                                           \
        N,                                                                     \
        K,                                                                     \
        fold_len,                                                              \
        global_row,                                                            \
        global_col,                                                            \
        local_row,                                                             \
        local_col,                                                             \
        float_len,                                                             \
        stride_k,                                                              \
        maxWGSize,                                                             \
        SIMD);                                                                 \
                                                                               \
    sycl::range<2> local_range{local_row, local_col};                          \
    sycl::range<2> global_range{                                               \
        global_row * local_row, global_col * local_col};                       \
                                                                               \
    sycl_kernel_submit(                                                        \
        global_range, local_range, at::xpu::getCurrentSYCLQueue(), kfn);       \
  }

#define WEIGHT_PACK_KERNEL(vec_len)                                       \
  {                                                                       \
    using vec_t = at::native::memory::aligned_vector<in_t, vec_len>;      \
    constexpr int align_bytes = alignof(vec_t);                           \
    int in_start = ((uint64_t)weight_data) % align_bytes / sizeof(in_t);  \
    if (in_start == 0 && K % vec_len == 0) {                              \
      VEC_WEIGHT_PACK_KERNEL_FUNC(vec_len, vec_t);                        \
    } else {                                                              \
      VEC_WEIGHT_PACK_KERNEL_FUNC(vec_len, in_t);                         \
    }                                                                     \
  }

  switch (float_len) {
    case 1: {
      WEIGHT_PACK_KERNEL(1);
      break;
    }
    case 2: {
      WEIGHT_PACK_KERNEL(2);
      break;
    }
    case 4: {
      WEIGHT_PACK_KERNEL(4);
      break;
    }
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Unexpected vectorization size for weight_to_int4pack. vec size ",
          float_len);
  }
}

} // namespace at::native::xpu