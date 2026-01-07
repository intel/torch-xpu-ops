/*
 * Copyright 2020-2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */
#include <ATen/native/xpu/sycl/ForeachCopyKernels.h>
#include <ATen/native/xpu/sycl/ForeachFunctors.h>
#include <ATen/native/xpu/sycl/MultiTensorApply.h>

namespace at::native::xpu {

namespace {

template <typename dst_t>
constexpr bool is_complex_dtype() {
  return std::is_same_v<dst_t, c10::complex<float>> ||
      std::is_same_v<dst_t, c10::complex<double>>;
}

template <typename dst_t, typename src_t>
constexpr int64_t compute_kILP() {
  constexpr size_t src_size = sizeof(src_t);
  constexpr size_t dst_size = sizeof(dst_t);
  constexpr size_t max_size = std::max(src_size, dst_size);

  // Adjust ILP based on data size to maintain good memory bandwidth
  if (max_size <= 2) {
    return kILP * 2;
  } else if (max_size <= 4) {
    return kILP;
  } else if (max_size <= 8) {
    return kILP / 2;
  } else {
    return 1;
  }
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
  static constexpr int64_t kILP = compute_kILP<dst_t, src_t>();

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
    const size_t tensor_loc = tlWGMeta[group_idx].wg_to_tensor;
    const size_t chunk_idx = tlWGMeta[group_idx].wg_to_chunk;
    const size_t chunk_offset = chunk_idx * chunk_size;
    const size_t n = tlAddress[tensor_loc].numel_to_tensor - chunk_offset;

    const size_t updated_chunk_size =
        std::min(static_cast<size_t>(chunk_size), n);

    src_t* src_ptr =
        static_cast<src_t*>(tlAddress[tensor_loc].addresses[0]) + chunk_offset;
    dst_t* dst_ptr =
        static_cast<dst_t*>(tlAddress[tensor_loc].addresses[1]) + chunk_offset;

    const bool all_aligned = is_aligned(src_ptr) && is_aligned(dst_ptr);
    constexpr bool same_sized_dtypes = sizeof(src_t) == sizeof(dst_t);

    src_t src_args[kILP];
    dst_t r_args[kILP];

    // vec path
    if (same_sized_dtypes && updated_chunk_size % kILP == 0 && all_aligned) {
      for (size_t i = item_idx; i * kILP < updated_chunk_size;
           i += item_range) {
        load_store<src_t, kILP>(src_args, src_ptr, 0, i);
#pragma unroll
        for (size_t ii = 0; ii < kILP; ++ii) {
          r_args[ii] = op(src_args[ii]);
        }
        load_store<dst_t, kILP>(dst_ptr, r_args, i, 0);
      }
      // non-vec path
    } else {
      for (size_t i = 0; i < updated_chunk_size; i += item_range * kILP) {
        const size_t base_idx = i + item_idx;
#pragma unroll
        for (size_t ii = 0; ii < kILP; ++ii) {
          const size_t i_start = base_idx + ii * item_range;
          if (i_start < updated_chunk_size) {
            dst_ptr[i_start] = op(src_ptr[i_start]);
          }
        }
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
        AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            at::ScalarType::Bool,
            src[0].scalar_type(),
            "foreach_tensor_copy",
            [&]() {
              using src_t = scalar_t;
              multi_tensor_apply<2>(
                  tensor_lists,
                  CopyFunctor<
                      dst_t,
                      src_t,
                      /* depth */ 2,
                      /* r_args_depth */ 1,
                      /* res_arg_index */ 1>(),
                  Copy<dst_t, src_t>());
            });
      });
}

} // namespace at::native::xpu
