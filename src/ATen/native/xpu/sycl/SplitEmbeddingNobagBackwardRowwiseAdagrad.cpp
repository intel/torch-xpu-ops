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

#include <ATen/native/xpu/sycl/SplitEmbeddingNobagBackwardRowwiseAdagrad.h>

namespace at::native::xpu {

    template <typename info_pta_t, typename info_t, bool nobag>
    void SplitEmbeddingBackwardCountUniqueIndicesKernel<info_pta_t, info_t, nobag>
    ::operator()(const sycl::nd_item<1>& item) const {
        const auto threadIdx_x = item.get_local_id(0);
        const auto blockIdx_x = item.get_group(0);
        const auto blockDim_x = item.get_local_range(0);
        const auto gridDim_x = item.get_group_range(0);

        const int32_t num_runs = sorted_linear_indices_num_runs[0];
        const auto T = weights_placements.size(0);
        for (auto run_id = blockIdx_x * blockDim_x + threadIdx_x;
            run_id < num_runs;
            run_id += blockDim_x * gridDim_x) {
            // Obtain the associated table id of the run id
            const auto segment_start = sorted_linear_indices_cumulative_run_lengths[run_id];
            const auto info = reinterpret_cast<const info_t*>(&sorted_infos[0])[segment_start];
            const auto t = nobag ? (info % T) : (info >> info_B_num_bits);

            int32_t t_next = -1;
            const int32_t unique_count_offset = run_id + 1;
            if (unique_count_offset < num_runs) {
            const auto segment_start_next = sorted_linear_indices_cumulative_run_lengths[unique_count_offset];
            const auto info_next = reinterpret_cast<const info_t*>(&sorted_infos[0])[segment_start_next];
            t_next = nobag ? (info_next % T) : (info_next >> info_B_num_bits);
            }

            if (t != t_next) {
                const auto placement = static_cast<PlacementType>(weights_placements[t]);
                if (placement != PlacementType::MANAGED_CACHING) {
                    // Record num unique indices for PlacementType::DEVICE from unique_count_offset
                    xpuAtomicAdd(&dev_or_uvm_unique_indices[t], unique_count_offset);
                }
                if (t_next != -1) {
                    const auto placement_next = static_cast<PlacementType>(weights_placements[t_next]);
                    if (placement_next != PlacementType::MANAGED_CACHING) {
                        // Record num unique indices for PlacementType::DEVICE from unique_count_offset
                        xpuAtomicAdd(&dev_or_uvm_unique_indices[t_next], -unique_count_offset);
                    }
                }
            }
        }
    }

    Tensor split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_pt2_xpu_wrapper(
        const Tensor& grad_output,
        const Tensor& /*host_weights*/,
        const Tensor& dev_weights,
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        const Tensor& weights_offsets,
        const c10::SymInt D,
        const Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits,
        const Tensor& indices,
        const Tensor& offsets,
        const Tensor& lxu_cache_locations,
        const int64_t BT_block_size,
        const int64_t max_segment_length_per_warp,
        const bool stochastic_rounding,
        const int64_t info_B_num_bits,
        const int64_t info_B_mask_int64,
        const bool use_uniq_cache_locations,
        const bool use_homogeneous_placements,
        Tensor momentum1_host, 
        Tensor momentum1_dev, 
        Tensor momentum1_uvm, 
        Tensor momentum1_placements, 
        Tensor momentum1_offsets, 
        Tensor learning_rate_tensor, 
        double eps, 
        double weight_decay, 
        int64_t weight_decay_mode, 
        double max_norm){
            static auto op =
                torch::Dispatcher::singleton()
                    .findSchemaOrThrow("fbgemm::split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_xpu", "")
                    .typed<Tensor(
                            const Tensor& /*grad_output*/,
                            const Tensor& /*dev_weights*/,
                            const Tensor& /*uvm_weights*/,
                            const Tensor& /*lxu_cache_weights*/,
                            const Tensor& /*weights_placements*/,
                            const Tensor& /*weights_offsets*/,
                            const c10::SymInt /*D*/,
                            const Tensor& /*hash_size_cumsum*/,
                            const int64_t /*total_hash_size_bits*/,
                            const Tensor& /*indices*/,
                            const Tensor& /*offsets*/,
                            const Tensor& /*ssd_row_addrs or lxu_cache_locations*/,
                            const int64_t /*BT_block_size*/,
                            const int64_t /*max_segment_length_per_warp*/,
                            const bool /*stochastic_rounding*/,
                            const int64_t /*info_B_num_bits*/,
                            const int64_t /*info_B_mask_int64*/,
                            const bool /*use_uniq_cache_locations*/,
                            const bool /*use_homogeneous_placements*/,
                            Tensor,
                            Tensor,
                            Tensor,
                            Tensor,
                            Tensor,
                            double,
                            double,
                            int64_t,
                            double
                    )>();

            return op.call(
                grad_output,
                dev_weights,
                uvm_weights,
                lxu_cache_weights,
                weights_placements,
                weights_offsets,
                D,
                hash_size_cumsum,
                total_hash_size_bits,
                indices,
                offsets,
                lxu_cache_locations,
                BT_block_size,
                max_segment_length_per_warp,
                stochastic_rounding,
                info_B_num_bits,
                info_B_mask_int64,
                use_uniq_cache_locations,
                use_homogeneous_placements,
                momentum1_dev, 
                momentum1_uvm, 
                momentum1_placements, 
                momentum1_offsets, 
                learning_rate_tensor, 
                eps, 
                weight_decay, 
                weight_decay_mode, 
                max_norm
            );
    }

    template <
        typename emb_t,
        typename cache_t,
        int32_t kFixedMaxVecsPerThread,
        int32_t kThreadGroupSize = kThreadGroupSize,
        int32_t VEC_WIDTH,
        bool kUseVecBlocking
    >
    void split_rowwise_adagrad_table_update_kernel(
        at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits>& dev_weights,
        at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits>& uvm_weights,
        at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits>& lxu_cache_weights,
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>& weights_placements,
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>& weights_offsets,
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>& sorted_lxu_cache_locations,
        Vec4TAcc<cache_t>* grad_sum,
        Vec4TAcc<cache_t>* smem_grad_sum,
        Vec4TAcc<cache_t>* shared_weight_update_row,
        const bool stochastic_rounding,
        const PhiloxXpuState& stochastic_rounding_philox_args,
        const uint32_t run_id,
        const uint32_t cache_loc_run_id,
        const int32_t D,
        const int32_t t,
        const int64_t idx,
        const float global_weight_decay,
        const int32_t max_vecs_per_thread,
        at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>& momentum1_dev,
        at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>& momentum1_uvm,
        at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>& momentum1_placements,
        at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits>& momentum1_offsets,
        const sycl::nd_item<2>& item,
        float learning_rate = 0,
        float eps = 0,
        float weight_decay = 0.0,
        int64_t weight_decay_mode = 0,
        float max_norm = 0.0
    ) {
        using acc_t = at::acc_type<cache_t, true>;
        const auto threadIdx_x = item.get_local_id(1);
        const auto blockDim_x = item.get_local_range(1);
        const auto sg = item.get_sub_group();

        // Copy value to max_vecs to make max_vecs_per_thread known at compile time
        // when kUseVecBlocking == false
        const int32_t max_vecs =
            kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
        const int64_t weights_offset = weights_offsets[t];
        emb_t* __restrict__ weights {nullptr};
        cache_t* __restrict__ cache_weights {nullptr};
        int32_t D_emb = D;

        const auto weights_placement = static_cast<PlacementType>(weights_placements[t]);
        auto cache_idx = -1;
        if (weights_placement == PlacementType::DEVICE) {
            weights = &dev_weights[weights_offset + idx * D_emb];
        } else {
            weights = &uvm_weights[weights_offset + idx * D_emb];
        }
        if (weights_placement == PlacementType::MANAGED_CACHING) {
            /* const auto */ cache_idx = sorted_lxu_cache_locations[cache_loc_run_id];
            if (cache_idx != kCacheLocationMissing) {
            cache_weights = &lxu_cache_weights[cache_idx][0];
            }
        }
        acc_t* __restrict__ momentum1;
        const auto momentum1_placement = static_cast<PlacementType>(momentum1_placements[t]);
        const int64_t momentum1_offset = momentum1_offsets[t];
        if (momentum1_placement == PlacementType::DEVICE) {
            momentum1 = &momentum1_dev[momentum1_offset];
        } else {
            momentum1 = &momentum1_uvm[momentum1_offset];
        }

        // stochastic_rounding is false
        auto weight_row_template =
            WeightRow<emb_t, cache_t, acc_t>(
                weights,
                cache_weights,
                D,
                stochastic_rounding,
                &stochastic_rounding_philox_args,
                threadIdx_x + run_id * blockDim_x);

        [[maybe_unused]] constexpr auto enable_optimizer_offloading = false;

        
        acc_t g_local_sum_square = 0.0;
        
        if constexpr (kUseVecBlocking) {
            // max_vecs is not known at compile time
            for (int32_t vec = 0;
                vec < max_vecs &&
                (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                
            const float4* grad = &smem_grad_sum[d_vec].acc;
            auto gx = grad->x();
            auto gy = grad->y();
            auto gz = grad->z();
            auto gw = grad->w();
            if (weight_decay_mode == 1) {
                // L2 regularization
                Vec4TAcc<cache_t> weight = weight_row_template.load(d);
                gx += weight_decay * weight.acc.x();
                gy += weight_decay * weight.acc.y();
                gz += weight_decay * weight.acc.z();
                gw += weight_decay * weight.acc.w();
            }
            g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
        
            }
        
        } else {
            // kFixedMaxVecsPerThread is known at compile time
            #pragma unroll kFixedMaxVecsPerThread
            for (int32_t vec = 0;
                vec < kFixedMaxVecsPerThread
                    && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                
            const float4* grad = &grad_sum[vec].acc;
            auto gx = grad->x();
            auto gy = grad->y();
            auto gz = grad->z();
            auto gw = grad->w();
            if (weight_decay_mode == 1) {
                // L2 regularization
                Vec4TAcc<cache_t> weight = weight_row_template.load(d);
                gx += weight_decay * weight.acc.x();
                gy += weight_decay * weight.acc.y();
                gz += weight_decay * weight.acc.z();
                gw += weight_decay * weight.acc.w();
            }
            g_local_sum_square += gx * gx + gy * gy + gz * gz + gw * gw;
        
            }
        }

        // Define the rowwise adagrad optimizer state struct view
        struct [[maybe_unused]] OptimizerState {
            acc_t momentum;
        };

        acc_t group_sum = sycl::reduce_over_group(sg, g_local_sum_square, sycl::plus<acc_t>{});
        const acc_t g_avg_square = group_sum / static_cast<acc_t>(D);

        acc_t multiplier = 0.0;
        acc_t correction = 0.0;
        if (threadIdx_x == 0) {	
            auto new_sum_square_grads = g_avg_square;
        
            // Update the optimizer state.  Use optimizer state offloading only if 
            // SSD and if enabled by the user
            if (enable_optimizer_offloading) {
                // Fetch the pointer to the optimizer state along the cache row
                auto* optimizer = weight_row_template.template optimizer_state_ptr<OptimizerState>();
                new_sum_square_grads += optimizer->momentum;
                optimizer->momentum = new_sum_square_grads;
            
            } else {
                new_sum_square_grads += momentum1[idx];
                momentum1[idx] = new_sum_square_grads;
            }

            multiplier = learning_rate / (sqrtf(new_sum_square_grads) + eps);
            if (weight_decay_mode == 1) {
                // L2 regularization
                correction = 1.0 - multiplier * weight_decay;
            } else if (weight_decay_mode == 2 || weight_decay_mode == 5) {
                // Decoupled weight decay
                correction = 1.0 - learning_rate * weight_decay;
            } else {
                // default value
                correction = 1.0;
            }
        }

        multiplier = sycl::group_broadcast(sg, multiplier, 0);
        correction = sycl::group_broadcast(sg, correction, 0);

        if constexpr (kUseVecBlocking) {
            // max_vecs is not known at compile time
            for (int32_t vec = 0;
                vec < max_vecs &&
                (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                
            Vec4TAcc<cache_t> weight_new = weight_row_template.load(d);
            Vec4TAcc<cache_t>& grad = smem_grad_sum[d_vec];
            weight_new.mul_(global_weight_decay);
            
            weight_new.acc.x() = correction * weight_new.acc.x() - multiplier * grad.acc.x();
            weight_new.acc.y() = correction * weight_new.acc.y() - multiplier * grad.acc.y();
            weight_new.acc.z() = correction * weight_new.acc.z() - multiplier * grad.acc.z();
            weight_new.acc.w() = correction * weight_new.acc.w() - multiplier * grad.acc.w();

            weight_row_template.store(weight_new, d);

            }
        
        } else {
            // kFixedMaxVecsPerThread is known at compile time
            #pragma unroll kFixedMaxVecsPerThread
            for (int32_t vec = 0;
                vec < kFixedMaxVecsPerThread
                    && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                
            Vec4TAcc<cache_t> weight_new = weight_row_template.load(d);
            Vec4TAcc<cache_t>& grad = grad_sum[vec];

            weight_new.mul_(global_weight_decay);
            
            weight_new.acc.x() = correction * weight_new.acc.x() - multiplier * grad.acc.x();
            weight_new.acc.y() = correction * weight_new.acc.y() - multiplier * grad.acc.y();
            weight_new.acc.z() = correction * weight_new.acc.z() - multiplier * grad.acc.z();
            weight_new.acc.w() = correction * weight_new.acc.w() - multiplier * grad.acc.w();

            weight_row_template.store(weight_new, d);

            }
        }

        if (max_norm > 0.0) {
            assert(!(std::is_same_v<emb_t, uint8_t> && !cache_weights) && "not supported for uint8 yet");

            // compute weight norm
            acc_t weight_sum_square = 0.0;
            for (int32_t vec = 0;
                vec < max_vecs && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                ++vec) {
                const int32_t d = (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH;
                Vec4TAcc<cache_t> weight_new = weight_row_template.load(d);
                weight_sum_square
                    += weight_new.acc.x() * weight_new.acc.x()
                    + weight_new.acc.y() * weight_new.acc.y()
                    + weight_new.acc.z() * weight_new.acc.z()
                    + weight_new.acc.w() * weight_new.acc.w();
            }
            acc_t group_sum = sycl::reduce_over_group(sg, weight_sum_square, sycl::plus<acc_t>{});
            const acc_t weight_norm = sqrtf(group_sum);

            // scale by max_norm if weight_norm exceeds max_norm
            if (threadIdx_x == 0) {
                multiplier = weight_norm > max_norm ? max_norm / weight_norm : 1.0f;
            }
            multiplier = sycl::group_broadcast(sg, multiplier, 0);
            if (weight_norm > max_norm) {
                for (int32_t vec = 0;
                    vec < max_vecs && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                    ++vec) {
                    const int32_t d = (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH;
                    Vec4TAcc<cache_t> weight_new = weight_row_template.load(d);

                    weight_new.acc.x() *= multiplier;
                    weight_new.acc.y() *= multiplier;
                    weight_new.acc.z() *= multiplier;
                    weight_new.acc.w() *= multiplier;
                    weight_row_template.store(weight_new, d);
                }
            }
        }
    }

    template <
        typename emb_t,
        typename grad_t,
        typename cache_t,
        typename index_t,
        int32_t kFixedMaxVecsPerThread,
        int32_t kThreadGroupSize,
        bool kUseVecBlocking>
    void SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelCTAPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>
        ::operator()(const sycl::nd_item<2>& item) const{

        constexpr int VEC_WIDTH = 4;
        constexpr auto kIsInt8 = false; // This kernel is only instantiated for non-int8 types, but we keep the condition here in case we want to instantiate it for int8 in the future
        int32_t T = weights_offsets.size(0);
        const int32_t num_long_runs = num_long_run_ids[0];
        const auto warp_id = item.get_local_id(0);
        const auto lane_id = item.get_local_id(1);
        const auto threadIdx_x = item.get_local_id(1);
        const auto blockIdx_x = item.get_group(0);
        const auto gridDim_x = item.get_group_range(0);
        const auto blockDim_y = item.get_local_range(0);
        const auto sg = item.get_sub_group();

        // Copy value to max_vecs to make max_vecs_per_thread known at compile time
        // when kUseVecBlocking == false
        const int32_t max_vecs =
            kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
        auto* smem_grad_sum = reinterpret_cast<Vec4TAcc<cache_t>*>(
            syclex::get_work_group_scratch_memory()
        ) + warp_id * max_vecs * kThreadGroupSize;

        for (auto long_run_id = blockIdx_x; long_run_id < num_long_runs; long_run_id += gridDim_x) {
                // The first thread block in the really long run has run_id in long_run_ids
                // and the rest have the negative of its offset (see find_long_segments kernel).
                int32_t cta_rank_on_current_run = 0;
                int32_t current_run_id = long_run_ids[long_run_id];
                if (current_run_id < 0) {
                    cta_rank_on_current_run = -long_run_ids[long_run_id];
                    current_run_id = long_run_ids[long_run_id - cta_rank_on_current_run];
                }
                const int32_t run_length =
                    sorted_linear_indices_cumulative_run_lengths[current_run_id + 1] -
                    sorted_linear_indices_cumulative_run_lengths[current_run_id];
                // This computation must agree with how we compute num_ctas_for_run in
                // find_long_segments kernel!
                const int32_t num_ctas_on_current_run =
                    use_deterministic_algorithms ? 1 : div_round_up(run_length, max_segment_length_per_cta);


                const int64_t linear_index = sorted_linear_indices_run[current_run_id];
                const int32_t segment_start =
                    sorted_linear_indices_cumulative_run_lengths[current_run_id] +
                    cta_rank_on_current_run * max_segment_length_per_cta;
                const int32_t segment_end = std::min(
                    use_deterministic_algorithms ? INT_MAX : segment_start + max_segment_length_per_cta,
                    sorted_linear_indices_cumulative_run_lengths[current_run_id + 1]);
                const int32_t SL = segment_end - segment_start;

                // Note that with shared embedding tables we can have multiple tables
                // (i.e. different values of `t` sharing the same segment).
                const auto info_0 = sorted_infos[segment_start];
                int32_t t_0 = info_0 % T;

                int64_t hash_size = hash_size_cumsum[t_0];
                int64_t idx = linear_index - hash_size;

                const int32_t SL_per_warp = div_round_up(SL, static_cast<int32_t>(blockDim_y));
                const int32_t sl_start = SL_per_warp * warp_id;
                const int32_t sl_end = std::min(static_cast<int32_t>(SL_per_warp * (warp_id + 1)), SL);

                // Accumulate gradients (compute grad_sum)
                Vec4TAcc<cache_t> grad_sum[kFixedMaxVecsPerThread];
                constexpr int32_t kGroupVecWidth = kThreadGroupSize * VEC_WIDTH;
                const int32_t num_vecs = (D + kGroupVecWidth - 1) / kGroupVecWidth;

                compute_grad_sum_unweighted_nobag<
                grad_t,
                cache_t,
                kFixedMaxVecsPerThread,
                kThreadGroupSize,
                VEC_WIDTH,
                kUseVecBlocking>(
                    item,
                    grad_sum,
                    smem_grad_sum,
                    grad_output,
                    D,
                    T,
                    sorted_infos,
                    segment_start,
                    sl_start,
                    sl_end,
                    num_vecs
                );
                // Do shared memory reduction only if we used multiple warps.
                if (SL > SL_per_warp) {
                    item.barrier(sycl::access::fence_space::local_space);

                    
                    if (blockDim_y >= 32) {
                        if (warp_id < 16) {
                            for (int32_t vec = 0; vec < max_vecs && (vec * kThreadGroupSize + lane_id) * VEC_WIDTH < D; ++vec) {
                                const int32_t d_vec = (vec * kThreadGroupSize + lane_id);
                                smem_grad_sum[d_vec] = vec4_acc(
                                    smem_grad_sum[d_vec],
                                    smem_grad_sum[d_vec +
                                        16 * max_vecs * kThreadGroupSize]);
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                                
                    if (blockDim_y >= 16) {
                        if (warp_id < 8) {
                            for (int32_t vec = 0; vec < max_vecs && (vec * kThreadGroupSize + lane_id) * VEC_WIDTH < D; ++vec) {
                            const int32_t d_vec = (vec * kThreadGroupSize + lane_id);
                            smem_grad_sum[d_vec] = vec4_acc(
                                smem_grad_sum[d_vec],
                                smem_grad_sum[d_vec +
                                    8 * max_vecs * kThreadGroupSize]);
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                                
                    if (blockDim_y >= 8) {
                        if (warp_id < 4) {
                            for (int32_t vec = 0; vec < max_vecs && (vec * kThreadGroupSize + lane_id) * VEC_WIDTH < D; ++vec) {
                            const int32_t d_vec = (vec * kThreadGroupSize + lane_id);
                            smem_grad_sum[d_vec] = vec4_acc(
                                smem_grad_sum[d_vec],
                                smem_grad_sum[d_vec +
                                    4 * max_vecs * kThreadGroupSize]);
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                                
                    if (blockDim_y >= 4) {
                        if (warp_id < 2) {
                            for (int32_t vec = 0; vec < max_vecs && (vec * kThreadGroupSize + lane_id) * VEC_WIDTH < D; ++vec) {
                            const int32_t d_vec = (vec * kThreadGroupSize + lane_id);
                            smem_grad_sum[d_vec] = vec4_acc(
                                smem_grad_sum[d_vec],
                                smem_grad_sum[d_vec +
                                    2 * max_vecs * kThreadGroupSize]);
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (warp_id == 0) {
                        if constexpr (kUseVecBlocking) {
                            // max_vecs is not known at compile time
                            for (int32_t vec = 0;
                                vec < max_vecs &&
                                (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                                ++vec) {
                                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                                [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                                
                                            smem_grad_sum[d_vec] = vec4_acc(
                                                smem_grad_sum[d_vec],
                                                smem_grad_sum[d_vec + max_vecs * kThreadGroupSize]
                                            );
                                        
                            }
                        
                        } else {
                            // kFixedMaxVecsPerThread is known at compile time
                            #pragma unroll kFixedMaxVecsPerThread
                            for (int32_t vec = 0;
                                vec < kFixedMaxVecsPerThread
                                    && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                                ++vec) {
                                const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                                [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                                
                                            grad_sum[vec] = vec4_acc(
                                                smem_grad_sum[d_vec],
                                                smem_grad_sum[d_vec + max_vecs * kThreadGroupSize]
                                            );
                            }
                        }
                    }
                }

                if (warp_id != 0) {
                    continue;
                }

                if (num_ctas_on_current_run > 1) {
                    int really_long_run_id = long_run_id_to_really_long_run_ids[long_run_id];
                    Vec4TAcc<cache_t> *temp_grad_accum_ptr =
                        reinterpret_cast<Vec4TAcc<cache_t>*>(&temp_grad_accum[really_long_run_id][0]);
                    
                    if constexpr (kUseVecBlocking) {
                        // max_vecs is not known at compile time
                        for (int32_t vec = 0;
                            vec < max_vecs &&
                            (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                            ++vec) {
                            const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                            
                                    xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.x(), smem_grad_sum[d_vec].acc.x());
                                    xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.y(), smem_grad_sum[d_vec].acc.y());
                                    xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.z(), smem_grad_sum[d_vec].acc.z());
                                    xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.w(), smem_grad_sum[d_vec].acc.w());
                        }
                    } else {
                        // kFixedMaxVecsPerThread is known at compile time
                        #pragma unroll kFixedMaxVecsPerThread
                        for (int32_t vec = 0;
                            vec < kFixedMaxVecsPerThread
                                && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                            ++vec) {
                                    const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                                    [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                            
                                    xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.x(), grad_sum[vec].acc.x());
                                    xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.y(), grad_sum[vec].acc.y());
                                    xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.z(), grad_sum[vec].acc.z());
                                    xpuAtomicAdd(&temp_grad_accum_ptr[d_vec].acc.w(), grad_sum[vec].acc.w());
                        }
                    }
                

                    int counter = 0;
                    if (threadIdx_x == 0) {
                        sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
                        counter = xpuAtomicAdd(&grad_accum_counter[really_long_run_id], -1);
                    }
                    counter = sycl::group_broadcast(sg, counter, 0);
                    // Only the thread block accumulated the gradient last does the weight update.
                    if (counter > 1) {
                        continue;
                    }
                    assert(counter == 1 && "Invalid grad_accum_counter. Race condition?");
                        
                    if constexpr (kUseVecBlocking) {
                        // max_vecs is not known at compile time
                        for (int32_t vec = 0;
                            vec < max_vecs &&
                            (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                            ++vec) {
                            const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                            [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                            
                                    smem_grad_sum[d_vec] = temp_grad_accum_ptr[d_vec];
                        }
                    
                    } else {
                        // kFixedMaxVecsPerThread is known at compile time
                        #pragma unroll kFixedMaxVecsPerThread
                        for (int32_t vec = 0;
                            vec < kFixedMaxVecsPerThread
                                && (kThreadGroupSize * vec + threadIdx_x) * VEC_WIDTH < D;
                            ++vec) {
                                    const int32_t d_vec = vec * kThreadGroupSize + threadIdx_x;
                                    [[maybe_unused]] const int32_t d = d_vec * VEC_WIDTH;
                                    
                                    grad_sum[vec] = temp_grad_accum_ptr[d_vec];
                        }
                    }
                }
                
                split_rowwise_adagrad_table_update_kernel<
                emb_t,
                cache_t,
                kFixedMaxVecsPerThread,
                kThreadGroupSize,
                VEC_WIDTH,
                kUseVecBlocking>(
                    dev_weights,
                    uvm_weights,
                    lxu_cache_weights,
                    weights_placements,
                    weights_offsets,
                    sorted_lxu_cache_locations,
                    grad_sum,
                    kUseVecBlocking ? smem_grad_sum : nullptr,
                    kIsInt8 ? smem_grad_sum : nullptr,
                    stochastic_rounding,
                    stochastic_rounding_philox_args,
                    current_run_id,
                    segment_start,
                    D,
                    t_0,
                    idx,
                    1, // global_weight_decay
                    max_vecs,
                    momentum1_dev, 
                    momentum1_uvm, 
                    momentum1_placements, 
                    momentum1_offsets, 
                    item,
                    learning_rate, 
                    eps, 
                    weight_decay, 
                    weight_decay_mode, 
                    max_norm
                );
        } // for each run
    }

    template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    typename index_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    bool kUseVecBlocking>
    void SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelWarpPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>
    ::operator()(const sycl::nd_item<2>& item) const{

        const auto threadIdx_y = item.get_local_id(0);
        const auto blockIdx_x = item.get_group(0);
        const auto gridDim_x = item.get_group_range(0);
        const auto blockDim_y = item.get_local_range(0);

        int32_t T = weights_offsets.size(0);
        const auto start_run_id = blockIdx_x * blockDim_y + threadIdx_y;

        constexpr int VEC_WIDTH = 4;

        const int32_t grad_sum_stride = max_D / VEC_WIDTH;
        auto* smem_grad_sum = kUseVecBlocking
        ? reinterpret_cast<Vec4TAcc<cache_t>*>(
            syclex::get_work_group_scratch_memory()
        )  + threadIdx_y * grad_sum_stride
        : nullptr;

        for (uint32_t run_id = start_run_id;
            run_id < sorted_linear_indices_run.size(0) && run_id < sorted_linear_indices_num_runs[0];
                run_id += gridDim_x * blockDim_y) {

            const int64_t linear_index = sorted_linear_indices_run[run_id];
            const int32_t segment_start =
                sorted_linear_indices_cumulative_run_lengths[run_id];
            const int32_t segment_end =
                sorted_linear_indices_cumulative_run_lengths[run_id + 1];
            const int32_t SL = segment_end - segment_start;


            if (SL >= max_segment_length_per_warp) {
                continue;
            }

            // now, each segment corresponds to exactly one table `t` and row in
            // that table (`idx`). Thus, we can hoist out some of the book-keeping.
            const auto info_0 = sorted_infos[segment_start];
            int32_t t_0 = info_0 % T;

            int64_t hash_size = hash_size_cumsum[t_0];
            int64_t idx = linear_index - hash_size;

            const int32_t sl_start = 0;
            const int32_t sl_end = SL;
            Vec4TAcc<cache_t> grad_sum[kFixedMaxVecsPerThread];
            constexpr int32_t kGroupVecWidth = kThreadGroupSize * VEC_WIDTH;
            const int32_t num_vecs = (D + kGroupVecWidth - 1) / kGroupVecWidth;

            compute_grad_sum_unweighted_nobag<
            grad_t,
            cache_t,
            kFixedMaxVecsPerThread,
            kThreadGroupSize,
            VEC_WIDTH,
            kUseVecBlocking>(
                item,
                grad_sum,
                smem_grad_sum,
                grad_output,
                D,
                T,
                sorted_infos,
                segment_start,
                sl_start,
                sl_end,
                num_vecs
            );

            // Copy value to max_vecs to make max_vecs_per_thread known at compile time
            // when kUseVecBlocking == false
            const int32_t max_vecs =
                kUseVecBlocking ? max_vecs_per_thread : kFixedMaxVecsPerThread;
            split_rowwise_adagrad_table_update_kernel<
            emb_t,
            cache_t,
            kFixedMaxVecsPerThread,
            kThreadGroupSize,
            VEC_WIDTH,
            kUseVecBlocking>(
                dev_weights,
                uvm_weights,
                lxu_cache_weights,
                weights_placements,
                weights_offsets,
                sorted_lxu_cache_locations,
                grad_sum,
                smem_grad_sum,
                smem_grad_sum, // shared_weight_update_row (reuse smem_grad_sum)
                stochastic_rounding,
                stochastic_rounding_philox_args,
                run_id,
                segment_start,
                D,
                t_0,
                idx,
                1, // global_weight_decay
                max_vecs,
                momentum1_dev, 
                momentum1_uvm, 
                momentum1_placements, 
                momentum1_offsets,
                item,
                learning_rate, 
                eps, 
                weight_decay, 
                weight_decay_mode, 
                max_norm
            ); // if not dense and optimizer != "none"
        }
    }


    Tensor split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_xpu(
        const Tensor& grad_output,
        const Tensor& dev_weights,
        const Tensor& uvm_weights,
        const Tensor& lxu_cache_weights,
        const Tensor& weights_placements,
        const Tensor& weights_offsets,
        const c10::SymInt D_,
        const Tensor& hash_size_cumsum,
        const int64_t total_hash_size_bits,
        const Tensor& indices,
        const Tensor& offsets,
        const Tensor& lxu_cache_locations,
        const int64_t unused_,
        const int64_t max_segment_length_per_warp,
        const bool stochastic_rounding,
        const int64_t info_B_num_bits, // int32_t
        const int64_t info_B_mask_int64, // uint32_t
        const bool use_uniq_cache_locations,
        const bool use_homogeneous_placements,
        Tensor momentum1_dev,
        Tensor momentum1_uvm,
        Tensor momentum1_placements,
        Tensor momentum1_offsets,
        Tensor learning_rate_tensor,
        double eps,
        double weight_decay,
        int64_t weight_decay_mode,
        double max_norm
    ) {
        const int64_t D = D_.guard_int(__FILE__, __LINE__);
        // convert `learning rate` to float since `learning rate` is float in kernels
        TORCH_CHECK(learning_rate_tensor.is_cpu(), "learning_rate_tensor tensor needs to be on CPU. Ensure learning_rate_tensor is on CPU or contact FBGEMM team if you get this error.")
        const float learning_rate = learning_rate_tensor.item<float>();

        TENSORS_ON_SAME_SYCL_XPU_IF_NOT_OPTIONAL(
            dev_weights,
            uvm_weights,
            lxu_cache_weights,
            weights_placements,
            weights_offsets,
            hash_size_cumsum,
            indices,
            offsets,
            lxu_cache_locations,
            grad_output);

        auto aligned_grad_output = aligned_grad_output_tensor_for_xpu_backwards(grad_output);

        SYCL_DEVICE_GUARD(dev_weights);
        auto max_D = D;
        TORCH_CHECK_LE(max_D, 2048);
        // Set total_unique_indices to total num indices by default
        const auto total_unique_indices = indices.numel();

        // short-circuit if there are zero indices.
        if (indices.numel() == 0) {
            return Tensor();
        }
        int32_t T = weights_offsets.numel();

        TORCH_CHECK_GT(T, 0);
        // offsets = [B x T  + 1]
        const auto total_B = offsets.size(0) - 1;
        TORCH_CHECK_GT(total_B, 0);
        // Cast info_B_mask from int64_t to uint32_t
        const uint32_t info_B_mask = info_B_mask_int64;

        int max_shared_bytes = 64 << 10;

        int shared_kb = max_shared_bytes >> 10;

        int used_shared_kb = shared_kb;
        const int used_shared_bytes = used_shared_kb << 10;

        Tensor linear_indices, linear_indices_sorted, infos_sorted,
            sorted_linear_indices_run, sorted_linear_indices_run_lengths,
            sorted_linear_indices_num_runs,
            sorted_linear_indices_cumulative_run_lengths;
        std::tie(
            linear_indices,
            linear_indices_sorted,
            infos_sorted,
            sorted_linear_indices_run,
            sorted_linear_indices_run_lengths,
            sorted_linear_indices_num_runs,
            sorted_linear_indices_cumulative_run_lengths) =
            transpose_embedding_input(
                hash_size_cumsum,
                total_hash_size_bits,
                indices,
                offsets,
                true,
                std::optional<Tensor>(),
                info_B_num_bits,
                info_B_mask,
                total_unique_indices,
                false // is_index_select
            );
        Tensor lxu_cache_locations_sorted = lxu_cache_locations;
        Tensor table_unique_indices_offsets;

        if (lxu_cache_locations.size(0) > 0) {
          lxu_cache_locations_sorted = at::empty_like(lxu_cache_locations);
          // size_t temp_storage_bytes = 0;
          AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_xpu_1", [&] {
              auto sorted = at::sort(linear_indices, 0, false);
              linear_indices_sorted.copy_(std::get<0>(sorted));
              auto permutation = std::get<1>(sorted);
              lxu_cache_locations_sorted.copy_(lxu_cache_locations.index_select(0, permutation));
          });
        }

        table_unique_indices_offsets = at::zeros_like(weights_placements);

        AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_xpu_2", [&] {
            DISPATCH_EMB_GRAD_CACHE_TYPES(
                dev_weights.scalar_type(),
                aligned_grad_output.scalar_type(),
                lxu_cache_weights.scalar_type(),
                    "split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_xpu",
                [&] {
                    // early memory release
                    linear_indices.reset();
                    linear_indices_sorted.reset();
                    const auto grad_output_reshaped = aligned_grad_output;

                    auto grad_output_accessor = grad_output_reshaped.packed_accessor64<grad_t, 2, RestrictPtrTraits>();

                    PhiloxXpuState rng_engine_inputs{};
                    if (stochastic_rounding && !std::is_same_v<emb_t, float>) {
                        auto gen = at::xpu::detail::getDefaultXPUGenerator(); // XPU default generator
                        std::lock_guard<std::mutex> lock(gen.mutex());
                        auto* xpu_gen = at::check_generator<at::XPUGeneratorImpl>(gen);
                        auto [seed, offset] = xpu_gen->philox_engine_inputs(4); // reserve 4 randoms/thread unit
                        rng_engine_inputs = {seed, offset};
                    }

                    DISPATCH_OPTIMAL_KERNEL(max_D, [&] {

                        auto long_run_ids = at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
                        auto num_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));

                        const bool use_deterministic_algorithms = at::globalContext().deterministicAlgorithms();
                        const int max_segment_length_per_cta = use_deterministic_algorithms ? INT_MAX : 1024;

                        Tensor long_run_id_to_really_long_run_ids;
                        if (use_deterministic_algorithms) {
                            long_run_id_to_really_long_run_ids =
                                at::empty(0, sorted_linear_indices_run_lengths.options());
                        } else {
                            long_run_id_to_really_long_run_ids =
                                at::empty({indices.numel()}, sorted_linear_indices_run_lengths.options());
                        }


                        auto num_really_long_run_ids = at::zeros({1}, indices.options().dtype(at::kInt));
                        auto grad_accum_counter = at::empty(
                            use_deterministic_algorithms ? 0 : (indices.numel() / max_segment_length_per_cta),
                            indices.options().dtype(at::kInt));

                        size_t local_range = kMaxThreads;
                        size_t global_range = div_round_up(static_cast<size_t>(total_unique_indices), local_range) * local_range;

                        sycl_kernel_submit(
                          sycl::range<1>(global_range),
                          sycl::range<1>(local_range),
                          getCurrentSYCLQueue(),
                          SplitEmbeddingBackwardCodegenFindLongSegments(
                              sorted_linear_indices_num_runs.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                              sorted_linear_indices_run_lengths.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                              long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                              num_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                              long_run_id_to_really_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                              num_really_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                              grad_accum_counter.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                              max_segment_length_per_warp,
                              max_segment_length_per_cta,
                              use_deterministic_algorithms));

                        // A temp buffer to accumulate gradients with atomics.
                        auto temp_grad_accum = at::zeros(
                            {use_deterministic_algorithms ? 0 : grad_accum_counter.numel(), max_D},
                            aligned_grad_output.options().dtype(std::is_same<cache_t, double>::value ? at::kDouble : at::kFloat));

                        DISPATCH_PLACEHOLDER_TYPES(
                        "split_embedding_backward_rowwise_adagrad_exact_placeholder_type_kernel",
                        [&] {
                            // Compute shared memory size for cta_per_row
                            constexpr auto kCacheAccBytes = sizeof(at::acc_type<cache_t, true>);
                            int32_t num_cta_per_row_groups = kMaxThreads / kThreadGroupSize;
                            validate_local_mem_size(getCurrentSYCLQueue(), used_shared_bytes);
                            const size_t cta_per_row_smem_bytes = compute_num_groups_and_dynamic_smem_bytes(
                                &num_cta_per_row_groups,
                                [&] (int num_groups) {
                                return num_groups * kCacheAccBytes * 4 * kThreadGroupSize * max_vecs_per_thread;
                                },
                                used_shared_bytes
                            );

                            const int32_t cta_per_row_grid_size = std::min(
                                div_round_up(static_cast<int32_t>(total_unique_indices), static_cast<int32_t>(kMaxThreads)),
                                static_cast<int32_t>(get_max_work_groups_()));

                            const size_t c_local_x = kThreadGroupSize; 
                            const size_t c_local_y = num_cta_per_row_groups;
                            const size_t c_grid = cta_per_row_grid_size;

                            sycl_kernel_submit<SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelCTAPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>>(
                                sycl::range<2>(c_grid * c_local_y, c_local_x),
                                sycl::range<2>(c_local_y, c_local_x),
                                getCurrentSYCLQueue(),
                                cta_per_row_smem_bytes,
                                SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelCTAPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>(
                                        grad_output_accessor,
                                        dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                        uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                        lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                                        weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(), // if optimizer != "none"
                                        weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                        D,
                                        hash_size_cumsum.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                        sorted_linear_indices_run.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                        sorted_linear_indices_cumulative_run_lengths.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        num_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        infos_sorted.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                        lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        use_uniq_cache_locations,
                                        table_unique_indices_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        stochastic_rounding,
                                        rng_engine_inputs, // if not dense and optimizer != "none"
                                        long_run_id_to_really_long_run_ids.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        temp_grad_accum.packed_accessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits>(),
                                        grad_accum_counter.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        max_segment_length_per_cta,
                                        use_deterministic_algorithms,
                                        max_vecs_per_thread,
                                        momentum1_dev.packed_accessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>(),
                                        momentum1_uvm.packed_accessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>(),
                                        momentum1_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        momentum1_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                        learning_rate,
                                        eps,
                                        weight_decay,
                                        weight_decay_mode,
                                        max_norm
                                    ));

                            // Compute shared memory size for warp_per_row
                            int32_t num_warp_per_row_groups = kBackwardMaxThreads / kThreadGroupSize;
                            int32_t warp_per_row_smem_bytes = 0;

                            if constexpr (kUseVecBlocking) {
                              warp_per_row_smem_bytes = compute_num_groups_and_dynamic_smem_bytes(
                                  &num_warp_per_row_groups,
                                  // Use max_D to compute shmem_bytes (for smem_grad_sum)
                                  // instead of using kMaxVecsPerThread to minimize the
                                  // shared memory allocation
                                  [&] (int32_t num_groups) {
                                      return max_D * num_groups * kCacheAccBytes;
                                  },
                                  used_shared_bytes);
                            }

                            int32_t warp_per_row_grid_size = std::min(
                                div_round_up(static_cast<int32_t>(total_unique_indices), static_cast<int32_t>(num_warp_per_row_groups)),
                                    static_cast<int32_t>(get_max_work_groups_()));

                            const size_t w_local_x = kThreadGroupSize; 
                            const size_t w_local_y = num_warp_per_row_groups;
                            const size_t w_grid = warp_per_row_grid_size;
                            
                            sycl_kernel_submit<SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelWarpPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>>(
                                sycl::range<2>(
                                  w_grid * w_local_y, w_local_x),
                                sycl::range<2>(
                                  w_local_y, w_local_x),
                                getCurrentSYCLQueue(),
                                warp_per_row_smem_bytes,
                                SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelWarpPerRow<emb_t, grad_t, cache_t, index_t, kFixedMaxVecsPerThread, kThreadGroupSize, kUseVecBlocking>(
                                        grad_output_accessor,
                                        dev_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                        uvm_weights.packed_accessor64<emb_t, 1, RestrictPtrTraits>(),
                                        lxu_cache_weights.packed_accessor64<cache_t, 2, RestrictPtrTraits>(),
                                        weights_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(), // if optimizer != "none"
                                        weights_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                        D,
                                        hash_size_cumsum.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                        sorted_linear_indices_run.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
                                        sorted_linear_indices_cumulative_run_lengths.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        infos_sorted.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                        lxu_cache_locations_sorted.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        use_uniq_cache_locations,
                                        table_unique_indices_offsets.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        sorted_linear_indices_num_runs.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        max_segment_length_per_warp,
                                        stochastic_rounding,
                                        rng_engine_inputs, // if not dense and optimizer != "none"
                                        max_D,
                                        max_vecs_per_thread,
                                        momentum1_dev.packed_accessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>(),
                                        momentum1_uvm.packed_accessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits>(),
                                        momentum1_placements.packed_accessor32<int32_t, 1, RestrictPtrTraits>(),
                                        momentum1_offsets.packed_accessor32<int64_t, 1, RestrictPtrTraits>(),
                                        learning_rate,
                                        eps,
                                        weight_decay,
                                        weight_decay_mode,
                                        max_norm)
                            );
                        }); // DISPATCH_PLACEHOLDER_TYPES
                    }); // DISPATCH_OPTIMAL_KERNEL
            }); // DISPATCH_EMB_GRAD_CACHE_TYPES
        }); // AT_DISPATCH_INDEX_TYPES
        return Tensor();
    }
} // namespace at::native::xpu
