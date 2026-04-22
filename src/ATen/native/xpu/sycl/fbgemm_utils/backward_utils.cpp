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

#include <ATen/native/xpu/sycl/fbgemm_utils/backward_utils.h>

namespace fbgemm_utils {

    Tensor asynchronous_complete_cumsum_xpu(const Tensor& t_in) {
        TORCH_CHECK(t_in.is_contiguous());
        TORCH_CHECK(t_in.dtype() == at::kInt || t_in.dtype() == at::kLong);
        TORCH_CHECK(t_in.dim() == 1 || t_in.dim() == 2);

        if (t_in.dim() == 1) {
            Tensor t_out = at::zeros({t_in.numel() + 1}, t_in.options());
            auto r_out = t_out.slice(0, 1);
            at::cumsum_out(r_out, t_in, 0);
            return t_out;
        }

        Tensor t_out = at::zeros({t_in.size(0), t_in.size(1) + 1}, t_in.options());
        auto r_out = t_out.slice(1, 1);
        at::cumsum_out(r_out, t_in, 1);
        return t_out;
    }

    // Pass 1: Mark run starts
    template <typename index_t>
    MarkRunStartsKernel<index_t>::MarkRunStartsKernel(
            const index_t* _sorted_input,
            int32_t* _run_starts,
            int64_t _total_elements)
            : sorted_input(_sorted_input),
            run_starts(_run_starts),
            total_elements(_total_elements) {}

    template <typename index_t>
    void MarkRunStartsKernel<index_t>::operator()(const sycl::nd_item<1>& item) const {
            const auto tid = item.get_global_id(0);
            
            if (tid >= total_elements) return;
            
            // Check if this is the start of a new run
            bool is_run_start = (tid == 0) || (sorted_input[tid] != sorted_input[tid - 1]);
            run_starts[tid] = is_run_start ? 1 : 0;
    }

    // Pass 2: Compact runs using prefix sum
    template <typename index_t>
    CompactRunsKernel<index_t>::CompactRunsKernel(
            const index_t* _sorted_input,
            const int32_t* _run_starts,
            const int32_t* _run_positions,  // prefix sum
            index_t* _unique_output,
            int32_t* _run_lengths,
            int64_t _total_elements)
            : sorted_input(_sorted_input),
            run_starts(_run_starts),
            run_positions(_run_positions),
            unique_output(_unique_output),
            run_lengths(_run_lengths),
            total_elements(_total_elements) {}

    template <typename index_t>
    void CompactRunsKernel<index_t>::operator()(const sycl::nd_item<1>& item) const {
        const auto tid = item.get_global_id(0);

        if (tid >= total_elements) {
            return;
        }

        if (run_starts[tid] == 1) {
            // Use the prefix-sum information in run_positions to locate the end of this run.
            // run_positions is a non-decreasing prefix sum over run_starts, so each increment
            // corresponds to a new run. For a run starting at tid with index run_idx, the next
            // run (if any) starts at the first position > tid where run_positions[pos] > run_idx.
            const int32_t run_idx = run_positions[tid];  // Position from prefix sum

            int64_t left = static_cast<int64_t>(tid) + 1;
            int64_t right = static_cast<int64_t>(total_elements);
            int64_t run_end = right;

            while (left < right) {
                int64_t mid = left + ((right - left) >> 1);
                if (run_positions[mid] > run_idx) {
                    run_end = mid;
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }

            int32_t run_length = static_cast<int32_t>(run_end - static_cast<int64_t>(tid));
            unique_output[run_idx] = sorted_input[tid];
            run_lengths[run_idx] = run_length;
        }
    }

    template <typename index_t, typename info_acc_t, bool nobag, bool vbe>
    LinearizeIndexKernel<index_t, info_acc_t, nobag, vbe>::LinearizeIndexKernel(
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
            FixedDivisor _fd)
            : hash_size_cumsum(_hash_size_cumsum),
                indices(_indices),
                offsets(_offsets),
                infos(_infos),
                linear_indices(_linear_indices),
                info_B_num_bits(_info_B_num_bits),
                info_B_mask(_info_B_mask),
                max_T(_max_T),
                max_B(_max_B),
                vbe_b_t_map(_vbe_b_t_map),
                fd(_fd) {}

    template <typename index_t, typename info_acc_t, bool nobag, bool vbe> 
    void LinearizeIndexKernel<index_t, info_acc_t, nobag, vbe>::operator()(const sycl::nd_item<1>& item) const{

        const auto threadIdx_x = item.get_local_id(0);
        const auto blockIdx_x = item.get_group(0);
        const auto blockDim_x = item.get_local_range(0);
        const auto sg = item.get_sub_group();
        
        // Print from first work-item only to avoid spam
        const auto T = hash_size_cumsum.size(0) - 1;

        auto b_t = blockIdx_x * blockDim_x + threadIdx_x;
        int32_t b;
        int32_t t;
        const auto total_B = offsets.size(0) - 1;
        bool valid = b_t < total_B;
        // info must be uint32_t (using auto will assign int32_t to info)
        uint32_t info = 0;

        if (vbe && valid) {
            info = vbe_b_t_map[b_t];
            reinterpret_cast<uint32_t*>(&t)[0] = info >> info_B_num_bits;
            reinterpret_cast<uint32_t*>(&b)[0] = info & info_B_mask;
        } else {
            fd.DivMod(b_t, &t, &b);
        }

        const index_t hash_offset = valid ? hash_size_cumsum[t] : -1;
        const auto indices_start = valid ? offsets[b_t] : -1;
        const auto L = valid ? offsets[b_t + 1] - indices_start : 0;
        const uint32_t lane_id = threadIdx_x % kThreadGroupSize;

        // Compile-time conditional
        if (nobag) {
            for (int32_t j = 0; j < kThreadGroupSize; ++j) {
            const auto indices_start_warp = sycl::select_from_group(sg, indices_start, j);
            const auto t_warp = sycl::select_from_group(sg, t, j);
            const auto L_warp = sycl::select_from_group(sg, L, j);
            const index_t hash_offset_warp = sycl::select_from_group(sg, hash_offset, j);
            
            for (auto i = lane_id; i < L_warp; i += kThreadGroupSize) {
                const index_t idx = indices[indices_start_warp + i];
                const auto l_t = (indices_start_warp + i) * T + t_warp;
                infos[indices_start_warp + i] = l_t;
                linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
            }
            }
        } else {
            // Store t in upper (32 - DEFAULT_INFO_B_NUM_BITS).
            // Store b in lower (DEFAULT_INFO_B_NUM_BITS).
            if (!vbe && valid) {
            info = (reinterpret_cast<uint32_t*>(&t)[0] << info_B_num_bits) |
                reinterpret_cast<uint32_t*>(&b)[0];
            }
            for (int32_t j = 0; j < kThreadGroupSize; ++j) {
            const auto indices_start_warp = sycl::select_from_group(sg, indices_start, j);
            const auto info_warp = sycl::select_from_group(sg, info, j);
            const auto L_warp = sycl::select_from_group(sg, L, j);
            const index_t hash_offset_warp = sycl::select_from_group(sg, hash_offset, j);
            for (int32_t i = lane_id; i < L_warp; i += kThreadGroupSize) {
                const index_t idx = indices[indices_start_warp + i];
                reinterpret_cast<uint32_t*>(&infos[0])[indices_start_warp + i] =
                    info_warp;
                linear_indices[indices_start_warp + i] = hash_offset_warp + idx;
            }
            }
        }
    }

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
        bool nobag,
        const std::optional<Tensor>& vbe_b_t_map,
        const int64_t info_B_num_bits,
        const int64_t info_B_mask,
        const int64_t total_unique_indices,
        const bool is_index_select,
        const std::optional<Tensor>& total_L_offsets,
        const int64_t fixed_L_per_warp,
        const int64_t num_warps_per_feature) {
        const bool vbe = vbe_b_t_map.has_value();
    
        TORCH_CHECK(nobag || !vbe || info_B_num_bits > 0);
        TORCH_CHECK(!vbe || info_B_mask > 0);
        TORCH_CHECK(
            !is_index_select || (fixed_L_per_warp > 0 && num_warps_per_feature > 0));

        const auto T = hash_size_cumsum.size(0) - 1;
        const auto total_B =
            !is_index_select ? (offsets.size(0) - 1) : (num_warps_per_feature * T);

        TORCH_CHECK(
            !is_index_select ||
            (total_L_offsets.has_value() &&
            total_L_offsets.value().numel() == T + 1));

        auto infos = at::empty_like(
            indices,
            indices.options().dtype(
                (nobag || is_index_select) ? at::kLong : at::kInt));
        auto infos_sorted = at::empty_like(infos);
        auto linear_indices = at::empty_like(indices);
        auto linear_indices_sorted = at::empty_like(indices);

        Tensor sorted_linear_indices_run;
        Tensor sorted_linear_indices_run_lengths;
        Tensor sorted_linear_indices_num_runs;


        AT_DISPATCH_INDEX_TYPES(
        infos.scalar_type(), "transpose_embedding_input_1", [&] {
            AT_DISPATCH_INDEX_TYPES(
                indices.scalar_type(), "transpose_embedding_input_2", [&] {
                if (!is_index_select) {
                    if (!nobag) {
                    TORCH_CHECK(false, "linearize_index_kernel kernel for bag operations not implemented yet in SYCL backend");
                    } else {
                        size_t local_range = kMaxThreads;
                        size_t global_range = div_round_up(static_cast<size_t>(total_B), local_range) * local_range;

                        sycl_kernel_submit<LinearizeIndexKernel<index_t, int64_t, true, false>>(
                        sycl::range<1>(global_range),
                        sycl::range<1>(local_range),
                        getCurrentSYCLQueue(),
                        LinearizeIndexKernel<index_t, int64_t, true, false>(
                              hash_size_cumsum.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                              indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                              offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                              infos.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                              linear_indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                              info_B_num_bits,
                              info_B_mask,
                              (1u << (DEFAULT_INFO_NUM_BITS - info_B_num_bits)) - 1,
                              (1u << info_B_num_bits) - 1,
                              nullptr,
                              FixedDivisor(total_B / T)));

                    }
                } else {
                    TORCH_CHECK(false, "linearize_index_index_select_kernel kernel not implemented yet in SYCL backend");
                }
                {
                    auto sort_result = at::sort(linear_indices, /*dim=*/0, /*descending=*/false);
                    auto sorted_keys = std::get<0>(sort_result);
                    auto perm = std::get<1>(sort_result);        // permutation indices (int64)

                    linear_indices_sorted.copy_(sorted_keys);
                    infos_sorted.copy_(infos.index_select(0, perm));
                }
                if (total_unique_indices != -1) {
                    TORCH_CHECK(total_unique_indices >= 0);
                    sorted_linear_indices_run =
                        at::empty({total_unique_indices}, indices.options());
                    sorted_linear_indices_run_lengths = at::zeros(
                        {total_unique_indices}, indices.options().dtype(at::kInt));
                } else {
                    sorted_linear_indices_run = at::empty_like(indices);
                    sorted_linear_indices_run_lengths =
                        at::zeros_like(indices, indices.options().dtype(at::kInt));
                }
                sorted_linear_indices_num_runs =
                    at::zeros({1}, indices.options().dtype(at::kInt));

                {
                    // Run-length encoding using custom SYCL kernel
                    const int64_t n = linear_indices_sorted.numel();
                    
                    // Temporary buffer to mark run starts (1 for start, 0 otherwise)
                    auto run_starts = at::zeros({n}, indices.options().dtype(at::kInt));
                    
                    // Pass 1: Mark run starts
                    size_t local_range = kMaxThreads;
                    size_t global_range = div_round_up(static_cast<size_t>(n), local_range) * local_range;
                    
                    sycl_kernel_submit<MarkRunStartsKernel<index_t>>(
                        sycl::range<1>(global_range),
                        sycl::range<1>(local_range),
                        getCurrentSYCLQueue(),
                        MarkRunStartsKernel<index_t>(
                                linear_indices_sorted.data_ptr<index_t>(),
                                run_starts.data_ptr<int32_t>(),
                                n));

                    // Compute prefix sum to get positions for each run start
                    // cumsum gives us [1, 1, 1, 2, 2, 3, ...] for positions [0, 1, 2, 3, 4, 5, ...]
                    // Subtract 1 to get 0-indexed positions: [0, 0, 0, 1, 1, 2, ...]
                    auto run_positions = at::cumsum(run_starts, /*dim=*/0, /*dtype=*/at::kInt).sub_(1);
                    
                    // Get total number of runs (last position + 1)
                    int32_t total_runs = run_positions[-1].item<int32_t>() + 1;
                    sorted_linear_indices_num_runs.fill_(total_runs);
                    
                    // Pass 2: Compact runs using the computed positions
                    sycl_kernel_submit<CompactRunsKernel<index_t>>(
                        sycl::range<1>(global_range),
                        sycl::range<1>(local_range),
                        getCurrentSYCLQueue(),
                        CompactRunsKernel<index_t>(
                                linear_indices_sorted.data_ptr<index_t>(),
                                run_starts.data_ptr<int32_t>(),
                                run_positions.data_ptr<int32_t>(),
                                sorted_linear_indices_run.data_ptr<index_t>(),
                                sorted_linear_indices_run_lengths.data_ptr<int32_t>(),
                                n));
                }
            });
        });

        auto sorted_linear_indices_cumulative_run_lengths =
            asynchronous_complete_cumsum_xpu(sorted_linear_indices_run_lengths);

        return {
            linear_indices,
            linear_indices_sorted,
            infos_sorted,
            sorted_linear_indices_run,
            sorted_linear_indices_run_lengths,
            sorted_linear_indices_num_runs,
            sorted_linear_indices_cumulative_run_lengths};
    }
} // namespace fbgemm_utils
