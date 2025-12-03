#include <ATen/Dispatch.h>
#include <comm/xpu_aten.h>

#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

#include <ATen/native/xpu/sycl/ForeachCopyKernels.h>

namespace at::native::xpu {

namespace {

template <typename dst_t>
constexpr bool is_complex_dtype() {
  return std::is_same_v<dst_t, c10::complex<float>> ||
      std::is_same_v<dst_t, c10::complex<double>>;
}

template <typename dst_t, typename src_t>
struct Copy {
  dst_t operator()(const src_t& x) {
    if constexpr (is_complex_dtype<src_t>() && !is_complex_dtype<dst_t>()) {
      return static_cast<dst_t>(x.real());
    } else {
      return static_cast<dst_t>(x);
    }
  }
};

template <
    typename dst_t,
    typename src_t,
    int depth,
    int r_args_depth,
    int res_arg_index>
struct CopyFunctor {
  static_assert(
      depth == 2 && r_args_depth == 1 && res_arg_index == 1,
      "CopyFunctor only supports depth=2, r_args_depth=1, res_arg_index=1");
  template <typename TLA, typename TLW, typename Op>
  void operator()(
      const int64_t chunk_size,
      TLA tlAddress,
      TLW tlWGMeta,
      sycl::nd_item<1> item_id,
      Op op) const {
    const auto item_idx = item_id.get_local_id(0);
    const auto item_range = item_id.get_local_range(0);
    const auto group_idx = item_id.get_group(0);
    const int tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    const int chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    const int64_t n =
        tlAddress[tensor_loc].numel_to_tensor - chunk_idx * chunk_size;

    src_t* src_ptr = (src_t*)tlAddress[tensor_loc].addresses[0];
    src_ptr += chunk_idx * chunk_size;
    dst_t* dst_ptr = (dst_t*)tlAddress[tensor_loc].addresses[1];
    dst_ptr += chunk_idx * chunk_size;

    const bool all_aligned = is_aligned(src_ptr) && is_aligned(dst_ptr);

    src_t src_args[kILP];
    dst_t r_args[kILP];

    // vec path
    if (n % kILP == 0 && chunk_size % kILP == 0 && all_aligned) {
      for (int64_t i = item_idx; i * kILP < n && i * kILP < chunk_size;
           i += item_range) {
        load_store(src_args, src_ptr, 0, i);
#pragma unroll
        for (int64_t ii = 0; ii < kILP; ++ii) {
          r_args[ii] = static_cast<dst_t>(op(src_args[ii]));
        }
        load_store(dst_ptr, r_args, i, 0);
      }
      // non-vec path
    } else {
      for (int64_t i = 0; i < n && i < chunk_size; i += item_range * kILP) {
#pragma unroll
        for (int64_t ii = 0; ii < kILP; ++ii) {
          const int64_t i_start = i + item_idx + ii * item_range;
          src_args[ii] = src_t{};
          if (i_start < n && i_start < chunk_size) {
            src_args[ii] = src_ptr[i_start];
          }
        }
#pragma unroll
        for (int64_t ii = 0; ii < kILP; ++ii) {
          r_args[ii] = static_cast<dst_t>(op(src_args[ii]));
        }
        store_args(dst_ptr, r_args, i, chunk_size, n, item_idx, item_range);
      }
    }
  }
};

} // anonymous namespace

void foreach_copy_list_kernel_(TensorList self, TensorList src) {
  std::vector<std::vector<at::Tensor>> tensor_lists{src.vec(), self.vec()};

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      self[0].scalar_type(),
      "foreach_tensor_copy",
      [&]() {
        using dst_t = scalar_t;
        using dst_opmath_t = at::opmath_type<dst_t>;
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            at::ScalarType::Bool,
            src[0].scalar_type(),
            "foreach_tensor_copy",
            [&]() {
              using src_t = scalar_t;
              using src_opmath_t = at::opmath_type<src_t>;
              multi_tensor_apply<2>(
                  tensor_lists,
                  CopyFunctor<
                      dst_t,
                      src_t,
                      /* depth */ 2,
                      /* r_args_depth */ 1,
                      /* res_arg_index */ 1>(),
                  Copy<dst_opmath_t, src_opmath_t>());
            });
      });
}

} // namespace at::native::xpu
