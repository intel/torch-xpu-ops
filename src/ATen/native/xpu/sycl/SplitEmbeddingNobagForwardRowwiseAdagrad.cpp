/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

/*
 * BSD License
 * 
 * For FBGEMM software
 * 
 * Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 *  * Neither the name Facebook nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific
 *    prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <ATen/native/xpu/sycl/SplitEmbeddingNobagForwardRowwiseAdagrad.h>

namespace at::native::xpu {

    template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kThreadGroupSize>
    SplitEmbeddingNoBagCodegenForwardUnweightedKernel<emb_t, cache_t, output_t, use_lxu_cache, index_t, kThreadGroupSize>
    ::SplitEmbeddingNoBagCodegenForwardUnweightedKernel(
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
    ): dev_weights(_dev_weights),
       uvm_weights(_uvm_weights),
       lxu_cache_weights(_lxu_cache_weights),
       weights_placements(_weights_placements),
       weights_offsets(_weights_offsets),
       D(_D),
       fd_B(_fd_B),
       indices(_indices),
       offsets(_offsets),
       lxu_cache_locations(_lxu_cache_locations),
       lxu_cache_conflict_misses(_lxu_cache_conflict_misses),
       output(_output) {}
    
    template <
    typename emb_t,
    typename cache_t,
    typename output_t,
    bool use_lxu_cache,
    typename index_t,
    size_t kThreadGroupSize>
    void SplitEmbeddingNoBagCodegenForwardUnweightedKernel<emb_t, cache_t, output_t, use_lxu_cache, index_t, kThreadGroupSize>
    ::operator()(const sycl::nd_item<2>& item) const {

        const auto threadIdx_x = item.get_local_id(1);
        const auto threadIdx_y = item.get_local_id(0);
        const auto blockIdx_x = item.get_group(0);
        const auto blockDim_y = item.get_local_range(0);
        const auto gridDim_x = item.get_group_range(0);
        const auto sg = item.get_sub_group();

        // Determine the linearized warp ID, and exit early if needed
        const auto total_B = offsets.size(0) - 1;
        // Since we place a limit on the grid size, we need to perform grid-striding
        for (auto b_t = blockIdx_x * blockDim_y + threadIdx_y; b_t < total_B; b_t += blockDim_y * gridDim_x) {

            // Determine the Table and Training Example IDs
            int32_t t;  // Table ID
            int32_t b;  // Training Example ID
            fd_B.DivMod(b_t, &t, &b);

            // Determine the number of indices Vec4(pooling factor) to look up within the bag
            overflow_safe_int_t indices_start = offsets[b_t];
            int32_t L = offsets[b_t + 1] - indices_start;

            // Get the offsets of the embedding dimensions of the tables and determine D

            // From the Table ID, fetch its weight tensor offset, locate that position
            // in the input weights tensor, and set the weights table pointer
            const auto weights_offset = weights_offsets[t];
            const emb_t* __restrict__ weights;
            const auto placement = static_cast<PlacementType>(weights_placements[t]);

            if (placement == PlacementType::DEVICE) {
                weights = &dev_weights[weights_offset];
            } else {
                weights = &uvm_weights[weights_offset];
            }

            // D is computed in the bag case or provided as function arg in the nobag case
            // (nobag only supports the case where the embedding dimensions are the same for all tables)
            int32_t D_emb = D;

            if constexpr (!use_lxu_cache) {
                // If use_lxu_cache is false, then the cache conflict miss rate is
                // effectively 100%
                // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
                for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
                    // Determine the L index that this thread will load data from in cooperative load
                    auto l = l_start + threadIdx_x;
                    // Cooperatively load the indices
                    const overflow_safe_int_t idx = l < L ? indices[indices_start + l] : 0;
                    // If idx is loaded
                    const auto offset_idx = idx * D_emb;
                    // Iterate over kThreadGroupSize indices
                    for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
                        // Load index from thread j in the group
                        [[maybe_unused]] auto offset_idx_j = sycl::select_from_group(sg, offset_idx, j);
                        overflow_safe_int_t output_j = indices_start + l_start + j;

                        const auto weights_row = WeightRowAccessor
                            <
                                emb_t,
                                cache_t
                            >(
                            &weights[offset_idx_j], // Load from the embedding table
                            D);
                        for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                            const auto d = i + threadIdx_x * VEC_WIDTH;
                            if (d < D) {
                                // Since there is no pooling, simply copy the weights to output
                                const auto weights_slice = weights_row.load(d);
                                // output is 2D
                                weights_slice.store(&output[output_j][d]);
                            }
                        }
                        
                    }
                }

            } else {
                if (placement != PlacementType::MANAGED_CACHING) {
                    // Load every row from HBM or UVM
                    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
                    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
                        // Determine the L index that this thread will load data from in cooperative load
                        auto l = l_start + threadIdx_x;
                        // Cooperatively load the indices
                        const overflow_safe_int_t idx = l < L ? indices[indices_start + l] : 0;
                        // If idx is loaded
                        const auto offset_idx = idx * D_emb;
                        // Iterate over kThreadGroupSize indices
                        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
                            // Load index from thread j in the group
                            [[maybe_unused]] auto offset_idx_j = sycl::select_from_group(sg, offset_idx, j);
                            overflow_safe_int_t output_j = indices_start + l_start + j;

                            const auto weights_row = WeightRowAccessor
                                <
                                    emb_t,
                                    cache_t
                                >(
                                &weights[offset_idx_j], // Load from the embedding table
                                D);
                            for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                                const auto d = i + threadIdx_x * VEC_WIDTH;
                                if (d < D) {
                                    // Since there is no pooling, simply copy the weights to output
                                    const auto weights_slice = weights_row.load(d);
                                    // output is 2D
                                    weights_slice.store(&output[output_j][d]);
                                }
                            }
                        }
                    }
                } else if (lxu_cache_conflict_misses && *lxu_cache_conflict_misses == 0) {
                    // If the UVM cache stats tensor is valid and tell us there are no
                    // conflict unique misses, then the miss rate is effectively 0%
                        
                    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
                    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
                        // Determine the L index that this thread will load data from in cooperative load
                        auto l = l_start + threadIdx_x;
                        // Cooperatively load the cache's indices
                        [[maybe_unused]] int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
                        // Iterate over kThreadGroupSize indices
                        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
                            overflow_safe_int_t output_j = indices_start + l_start + j;
                            // Load cache's index from thread j in the group
                            [[maybe_unused]] int32_t cache_idx_j
                                = use_lxu_cache ? sycl::select_from_group(sg, cache_idx, j) : 0;
                                
                            const cache_t* cache_weights = reinterpret_cast<const cache_t*>(
                                &lxu_cache_weights[cache_idx_j][0]);
                            const auto weights_row = WeightRowAccessor
                                <
                                    cache_t,
                                    cache_t
                                >(
                                cache_weights, // Load from the cache
                                D);
                            for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                                const auto d = i + threadIdx_x * VEC_WIDTH;
                                if (d < D) {
                                    // Since there is no pooling, simply copy the weights to output
                                    const auto weights_slice = weights_row.load(d);
                                    // output is 2D
                                    weights_slice.store(&output[output_j][d]);
                                }
                            }
                        }
                    }
                } else {
                    // Else, the cache conflict miss rate is mixed
        
                    
                    // Iterate over each kThreadGroupSize-sized subset of L indices in the bag
                    for (int32_t l_start = 0; l_start < L; l_start += kThreadGroupSize) {
                        // Determine the L index that this thread will load data from in cooperative load
                        auto l = l_start + threadIdx_x;
                        // Cooperatively load the indices
                        const overflow_safe_int_t idx = l < L ? indices[indices_start + l] : 0;
                        // If idx is loaded
                        const auto offset_idx = idx * D_emb;
                        // Cooperatively load the cache's indices
                        [[maybe_unused]] int32_t cache_idx = (use_lxu_cache && placement == PlacementType::MANAGED_CACHING && l < L) ? lxu_cache_locations[indices_start + l] : 0;
                        // Iterate over kThreadGroupSize indices
                        for (auto j = 0; j < kThreadGroupSize && l_start + j < L; ++j) {
                            // Load index from thread j in the group
                            [[maybe_unused]] auto offset_idx_j = sycl::select_from_group(sg, offset_idx, j); //SHFL_SYNC(offset_idx, j);
                            overflow_safe_int_t output_j = indices_start + l_start + j;
                            // Load cache's index from thread j in the group
                            [[maybe_unused]] int32_t cache_idx_j
                                = use_lxu_cache ? sycl::select_from_group(sg, cache_idx, j) : 0; //SHFL_SYNC(cache_idx, j);

                            if (placement == PlacementType::MANAGED_CACHING
                                && cache_idx_j != kCacheLocationMissing
                            ) {
                                const cache_t* cache_weights = reinterpret_cast<const cache_t*>(
                                    &lxu_cache_weights[cache_idx_j][0]);
                                const auto weights_row = WeightRowAccessor
                                    <
                                        cache_t,
                                        cache_t
                                    >(
                                    cache_weights, // Load from the cache
                                    D);
                                for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                                    const auto d = i + threadIdx_x * VEC_WIDTH;
                                    if (d < D) {
                                        // Since there is no pooling, simply copy the weights to output
                                        const auto weights_slice = weights_row.load(d);
                                        // output is 2D
                                        weights_slice.store(&output[output_j][d]);
                                    }
                                }
                            } else {
                                const auto weights_row = WeightRowAccessor
                                    <
                                        emb_t,
                                        cache_t
                                    >(
                                    &weights[offset_idx_j], // Load from the embedding table
                                    D);
                                for (int32_t i = 0; i < D; i += kThreadGroupSize * VEC_WIDTH) {
                                    const auto d = i + threadIdx_x * VEC_WIDTH;
                                    if (d < D) {
                                        // Since there is no pooling, simply copy the weights to output
                                        const auto weights_slice = weights_row.load(d);
                                        // output is 2D
                                        weights_slice.store(&output[output_j][d]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } // for b_t
    }

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
    ) {
        const int64_t D = D_.guard_int(__FILE__, __LINE__);

        TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            indices,
            offsets,
            lxu_cache_locations,
            dev_weights
        );

        int32_t total_L = indices.numel();
        int32_t T = weights_offsets.numel();
        TORCH_CHECK_GT(T, 0);
        // offsets = [B x T  + 1]
        const auto total_B = offsets.size(0) - 1;
        const int32_t B = total_B / T;
        TORCH_CHECK_GE(B, 0);
        TORCH_CHECK_GT(D, 0);
        TORCH_CHECK_EQ(D % 4, 0);

        Tensor output;
        SparseType o_dtype = static_cast<SparseType>(output_dtype);
        TORCH_CHECK(o_dtype == SparseType::FP32 || o_dtype == SparseType::FP16 ||
                    o_dtype == SparseType::BF16 || o_dtype == SparseType::INT8);

        int64_t adjusted_D = D;

        output = at::empty({total_L, adjusted_D}, dev_weights.options().dtype(getScalarType(o_dtype))); // if nobag

        if (B == 0) {
            return output;
        }


        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "batched_embedding_nobag_forward_kernel_1", [&] {
        DISPATCH_EMB_CACHE_OUTPUT_TYPES(
            dev_weights.scalar_type(),
            lxu_cache_weights.scalar_type(),
            output.scalar_type(),
            "batched_embedding_nobag_forward_kernel_2", [&] {
            // Check if LXU cache is used
            bool use_lxu_cache = lxu_cache_weights.numel() > 0;

            // dense_embedding_nobag_codegen_forward_unweighted_small_kernel <- Goes here when implemented, currently only the large kernel is implemented in SYCL backend

            DISPATCH_KERNEL_FOR_CACHE_CASE(use_lxu_cache, [&] {
                try {
                    sycl::queue& queue = c10::xpu::getCurrentXPUStream().queue();
                    auto device = queue.get_device();

                    const size_t local_x = kThreadGroupSize; 
                    const size_t local_y = kForwardMaxThreads / kThreadGroupSize;
                    const size_t grid_x = div_round_up(static_cast<size_t>(total_B), local_y);

                    queue.submit([&](sycl::handler& cgh) {
                        cgh.parallel_for<SplitEmbeddingNoBagCodegenForwardUnweightedKernel<emb_t, cache_t, output_t, use_cache_t, index_t, kThreadGroupSize>>(
                            sycl::nd_range<2>(
                                sycl::range<2>(grid_x * local_y, local_x),
                                sycl::range<2>(local_y, local_x)
                            ),
                            SplitEmbeddingNoBagCodegenForwardUnweightedKernel<emb_t, cache_t, output_t, use_cache_t, index_t, kThreadGroupSize>(
                                dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                                weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                D,
                                FixedDivisor(B),
                                indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                lxu_cache_locations.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                uvm_cache_stats.size(0) == 0
                                    ? nullptr
                                    : (uvm_cache_stats.data_ptr<int32_t>() + uvm_cache_stats_index::num_conflict_unique_misses),
                                output.packed_accessor64<output_t, 2, RestrictPtrTraits>()
                            )
                        );
                    });
                    
                } catch (const sycl::exception& e) {
                    std::cerr << "SYCL exception: " << e.what() << std::endl;
                    throw;
                }
            }); // if has_experimental
            });
        });
    return output;
    }

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
        ){
        static auto op =
            torch::Dispatcher::singleton()
                .findSchemaOrThrow("fbgemm::split_embedding_nobag_codegen_forward_unweighted_xpu", "")
                .typed<Tensor(
                    const Tensor& /*host_weights*/,
                    const Tensor& /*dev_weights*/,
                    const Tensor& /*uvm_weights*/,
                    const Tensor& /*lxu_cache_weights*/,
                    const Tensor& /*weights_placements*/,
                    const c10::SymInt /*D*/,
                    const Tensor& /*indices*/,
                    const Tensor& /*offsets*/,
                    const Tensor& /*row_addrs or lxu_cache_locations*/,
                    const Tensor& /*uvm_cache_stats_*/,
                    const int64_t /*output_dtype*/,
                    const bool
                )>();

        return op.call(
                dev_weights,
                uvm_weights,
                lxu_cache_weights,
                weights_placements,
                weights_offsets,
                D,
                indices,
                offsets,
                lxu_cache_locations,
                uvm_cache_stats,
                output_dtype, 
                is_experimental
        );
    };
} // namespace at::native::xpu
