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

#pragma once

#include <ATen/ATen.h>
#include <sycl/sycl.hpp>
#include <ATen/core/TensorAccessor.h>
#include <ATen/Operators.h>
#include <ATen/native/StridedRandomAccessor.h>

#include <ATen/native/xpu/sycl/fbgemm_utils/utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/vec4.h>

using Tensor = at::Tensor;
using at::native::RestrictPtrTraits;

namespace fbgemm_utils {

    Tensor asynchronous_complete_cumsum_xpu(const Tensor& t_in);

    // Pass 1: Mark run starts
    template <typename index_t>
    class MarkRunStartsKernel {
    public:
        MarkRunStartsKernel(
            const index_t* _sorted_input,
            int32_t* _run_starts,
            int64_t _total_elements);

        void operator()(const sycl::nd_item<1>& item) const;

    private:
        const index_t* sorted_input;
        int32_t* run_starts;
        int64_t total_elements;
    };

    // Pass 2: Compact runs using prefix sum
    template <typename index_t>
    class CompactRunsKernel {
    public:
        CompactRunsKernel(
            const index_t* _sorted_input,
            const int32_t* _run_starts,
            const int32_t* _run_positions,  // prefix sum
            index_t* _unique_output,
            int32_t* _run_lengths,
            int64_t _total_elements);

        void operator()(const sycl::nd_item<1>& item) const;

    private:
        const index_t* sorted_input;
        const int32_t* run_starts;
        const int32_t* run_positions;
        index_t* unique_output;
        int32_t* run_lengths;
        int64_t total_elements;
    };

    class SplitEmbeddingBackwardCodegenFindLongSegments {
    public:
        SplitEmbeddingBackwardCodegenFindLongSegments(
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _sorted_linear_indices_num_runs,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _sorted_linear_indices_run_lengths,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _long_run_ids,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _num_long_run_ids,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _long_run_id_to_really_long_run_ids,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _num_really_long_run_ids,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _grad_accum_counter,
            const int32_t _max_segment_length_per_warp,
            const int32_t _max_segment_length_per_cta,
            const bool _use_deterministic_algorithms)
            : sorted_linear_indices_num_runs(_sorted_linear_indices_num_runs), 
            sorted_linear_indices_run_lengths(_sorted_linear_indices_run_lengths), 
            long_run_ids(_long_run_ids), 
            num_long_run_ids(_num_long_run_ids), 
            long_run_id_to_really_long_run_ids(_long_run_id_to_really_long_run_ids), 
            num_really_long_run_ids(_num_really_long_run_ids), 
            grad_accum_counter(_grad_accum_counter), 
            max_segment_length_per_warp(_max_segment_length_per_warp), 
            max_segment_length_per_cta(_max_segment_length_per_cta), 
            use_deterministic_algorithms(_use_deterministic_algorithms) {};
        
        void operator()(const sycl::nd_item<1>& item) const {
            const auto threadIdx_x = item.get_local_id(0);
            const auto blockIdx_x = item.get_group(0);
            const auto blockDim_x = item.get_local_range(0);
            const auto gridDim_x = item.get_group_range(0);

            const int32_t num_runs = sorted_linear_indices_num_runs[0];

            for (auto run_id = blockIdx_x * blockDim_x + threadIdx_x; run_id < num_runs; run_id += blockDim_x * gridDim_x) {
                if (sorted_linear_indices_run_lengths[run_id] >= max_segment_length_per_warp) {
                    const int num_ctas_for_run =
                        use_deterministic_algorithms ? 1 : div_round_up(sorted_linear_indices_run_lengths[run_id], max_segment_length_per_cta);

                    const auto long_run_idx = xpuAtomicAdd(&num_long_run_ids[0], num_ctas_for_run);
                    long_run_ids[long_run_idx] = run_id;
                    for (int i = 1; i < num_ctas_for_run; ++i) {
                        long_run_ids[long_run_idx + i] = -i;
                    }
                    if (num_ctas_for_run > 1) {
                        const auto really_long_run_idx = xpuAtomicAdd(&num_really_long_run_ids[0], 1);
                        grad_accum_counter[really_long_run_idx] = num_ctas_for_run;
                        for (int i = 0; i < num_ctas_for_run; ++i) {
                            long_run_id_to_really_long_run_ids[long_run_idx + i] = really_long_run_idx;
                        }
                    }
                }
            }
        }

    private:
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_run_lengths;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_ids;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> num_long_run_ids;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_id_to_really_long_run_ids;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> num_really_long_run_ids;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> grad_accum_counter;
        int32_t max_segment_length_per_warp;
        int32_t max_segment_length_per_cta;
        bool use_deterministic_algorithms;
    };

    template <typename index_t, typename info_acc_t, bool nobag, bool vbe>
    class LinearizeIndexKernel {
        public:
            LinearizeIndexKernel(
                const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _hash_size_cumsum,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> _indices,
                const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> _offsets,
                at::PackedTensorAccessor32<info_acc_t, 1, RestrictPtrTraits> _infos,
                at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> _linear_indices,
                const int32_t _info_B_num_bits,
                const uint32_t _info_B_mask,
                const uint32_t _max_T,
                const uint32_t _max_B,
                // Use a raw pointer to avoid creating dummy PackedTensorAccessor
                const uint32_t* const __restrict__ _vbe_b_t_map,
                FixedDivisor _fd);
            
            void operator()(const sycl::nd_item<1>& item) const;

        private:
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets;
            mutable at::PackedTensorAccessor32<info_acc_t, 1, RestrictPtrTraits> infos;
            mutable at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> linear_indices;
            const int32_t info_B_num_bits;
            const uint32_t info_B_mask;
            const uint32_t max_T;
            const uint32_t max_B;
            const uint32_t* const __restrict__ vbe_b_t_map;
            FixedDivisor fd;
    };

      /**
     * "Transpose" embedding inputs by sorting indices by their values.
     * Logically this transpose compressed sparse row (CSR) representation
     * stored in indices and offsets to compressed sparse column (CSC).
     */
    std::tuple<
        Tensor /*linear_indices*/,
        Tensor /*linear_indices_sorted*/,
        Tensor /*infos_sorted*/,
        Tensor /*sorted_linear_indices_run*/,
        Tensor /*sorted_linear_indices_run_lengths*/,
        Tensor /*sorted_linear_indices_num_runs*/,
        Tensor /*sorted_linear_indices_cumulative_run_lengths*/>
    transpose_embedding_input(
        Tensor hash_size_cumsum,
        int64_t total_hash_size_bits,
        Tensor indices,
        Tensor offsets,
        bool nobag = false,
        const std::optional<Tensor>& vbe_b_t_map = std::optional<at::Tensor>(),
        const int64_t info_B_num_bits = 26,
        const int64_t info_B_mask = 0x2FFFFFF,
        const int64_t total_unique_indices = -1,
        const bool is_index_select = false,
        const std::optional<Tensor>& total_L_offsets = std::optional<at::Tensor>(),
        const int64_t fixed_L_per_warp = 0,
        const int64_t num_warps_per_feature = 0);

    template <
        typename grad_t,
        typename cache_t,
        int32_t kFixedMaxVecsPerThread,
        int32_t kThreadGroupSize,
        int32_t VEC_WIDTH,
        bool kUseVecBlocking
    >
    void compute_grad_sum_unweighted_nobag(
        const sycl::nd_item<2>& item,
        Vec4TAcc<cache_t>* grad_sum,
        Vec4TAcc<cache_t>* smem_grad_sum,
        const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits>& grad_output,
        const int32_t D,
        const int32_t T,
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>& sorted_infos,
        const int32_t segment_start,
        const int32_t sl_start,
        const int32_t sl_end,
        const int32_t num_vecs
    ) {
        const auto threadIdx_x = item.get_local_id(1);
        const auto sg = item.get_sub_group();

        // Copy value to vecs to make num_vecs known at compile time when
        // kUseVecBlocking == false
        const int32_t vecs = kUseVecBlocking ? num_vecs : kFixedMaxVecsPerThread;
        for (int32_t vec_start = 0;
            vec_start < vecs;
            vec_start += kFixedMaxVecsPerThread) {

            // Reset grad_sum vectors
            #pragma unroll kFixedMaxVecsPerThread
            for (int32_t vec = 0; vec < kFixedMaxVecsPerThread; vec++) {
                grad_sum[vec].acc.x() = 0;
                grad_sum[vec].acc.y() = 0;
                grad_sum[vec].acc.z() = 0;
                grad_sum[vec].acc.w() = 0;
            }

            for (int32_t sl = sl_start; sl < sl_end; sl += kThreadGroupSize) {
                auto sl_j = sl + threadIdx_x; // if not nobag
                int64_t l_t = sl_j < sl_end ? sorted_infos[segment_start + sl_j] : 0;
                int32_t l = l_t / T; // if not nobag
                for (int32_t j = 0; j < kThreadGroupSize && sl + j < sl_end; ++j) {
                    int32_t l_j = sycl::select_from_group(sg, l, j);
                
                    #pragma unroll kFixedMaxVecsPerThread
                    for (int32_t vec = 0; vec < kFixedMaxVecsPerThread && (((vec + vec_start) * kThreadGroupSize + threadIdx_x) * VEC_WIDTH) < D; ++vec) {
                            const int32_t d = (((vec + vec_start) * kThreadGroupSize + threadIdx_x) * VEC_WIDTH);
                            Vec4TAcc<grad_t> grad_out_vec(
                                &grad_output[l_j][d] // if nobag
                            );
                            grad_sum[vec].add_(grad_out_vec);
                    }
                }
            }

            if (smem_grad_sum) {
                // Store grad_sum in smem_grad_sum
                #pragma unroll kFixedMaxVecsPerThread
                for (int32_t vec = 0;
                    (vec < kFixedMaxVecsPerThread) && ((vec + vec_start) * kThreadGroupSize + threadIdx_x) * VEC_WIDTH < D;
                    ++vec) {
                    const int32_t d_vec = ((vec + vec_start) * kThreadGroupSize + threadIdx_x);
                    smem_grad_sum[d_vec] = grad_sum[vec];
                }
            }
        }
    } 
} // namespace fbgemm_utils
