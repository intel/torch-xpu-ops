/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <ATen/native/StridedRandomAccessor.h>

#include <ATen/native/xpu/sycl/fbgemm_utils/utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/weight_row.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/tensor_utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/split_embeddings_cache_xpu.h>

using Tensor = at::Tensor;

using namespace fbgemm_utils;
using at::native::RestrictPtrTraits;

namespace at::native::xpu {
    #define DISPATCH_KERNEL_FOR_CACHE_CASE(CACHE_CASE_, ...)                       \
    [&] {                                                                        \
        if (CACHE_CASE_ == false) {                                      \
        constexpr auto use_cache_t = false;                            \
        return __VA_ARGS__();                                                    \
        }                                                                          \
        if (CACHE_CASE_ == true) {                                      \
        constexpr auto use_cache_t = true;                            \
        return __VA_ARGS__();                                                    \
        }                                                                          \
        return;                                                                    \
    }()

    template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kThreadGroupSize>
    class SplitEmbeddingNoBagCodegenForwardUnweightedKernel {
        public:
            SplitEmbeddingNoBagCodegenForwardUnweightedKernel(
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> _dev_weights,
                const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> _uvm_weights,
                const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> _lxu_cache_weights,
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _weights_placements,
                const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _weights_offsets,
                int64_t _D,
                FixedDivisor _fd_B,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> _indices,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> _offsets,
                const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _lxu_cache_locations,
                const int32_t* _lxu_cache_conflict_misses,
                at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> _output
            );

            void operator()(const sycl::nd_item<2>& item) const;

        private:
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights;
            const at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights;
            const at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets;
            const int64_t D;
            FixedDivisor fd_B;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> lxu_cache_locations;
            const int32_t* lxu_cache_conflict_misses;
            mutable at::PackedTensorAccessor64<output_t, 2, RestrictPtrTraits> output;
    };

    Tensor split_embedding_nobag_codegen_forward_unweighted_xpu(
            const Tensor& dev_weights,
            const Tensor& uvm_weights,
            const Tensor& lxu_cache_weights,
            const Tensor& weights_placements,
            const Tensor& weights_offsets,
            const c10::SymInt D_,
            const Tensor& indices,
            const Tensor& offsets,
            const Tensor& lxu_cache_locations,
            const Tensor& uvm_cache_stats,
            const int64_t output_dtype,
            const bool is_experimental
        );

Tensor split_embedding_nobag_codegen_forward_unweighted_pt2_xpu_wrapper(
        const Tensor& /*host_weights*/,
        const Tensor& dev_weights,
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        const Tensor& weights_offsets,
        const c10::SymInt D,
        const Tensor& hash_size_cumsum,
        const Tensor& indices,
        const Tensor& offsets,
        const Tensor& lxu_cache_locations,
        const Tensor& uvm_cache_stats,
        const bool is_experimental,
        const int64_t output_dtype
        );
        
} // namespace fbgemm
