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

#include <sycl/sycl.hpp>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/csrc/autograd/record_function_ops.h>
#include <ATen/native/StridedRandomAccessor.h>
#include <ATen/xpu/XPUGeneratorImpl.h>

#include <ATen/native/xpu/sycl/fbgemm_utils/utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/tensor_utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/split_embeddings_cache_xpu.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/backward_utils.h>
#include <ATen/native/xpu/sycl/fbgemm_utils/weight_row.h>


using Tensor = at::Tensor;

using namespace fbgemm_utils;
using namespace fbgemm_utils::utils;
using at::native::RestrictPtrTraits;
namespace syclex = sycl::ext::oneapi::experimental;
using float4 = sycl::float4;

namespace at::native::xpu {
    #define DISPATCH_PLACEHOLDER_TYPES(NAME, ...) \
    return __VA_ARGS__();

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
        double eps = 0, 
        double weight_decay = 0.0, 
        int64_t weight_decay_mode = 0, 
        double max_norm = 0.0);

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
    );

    template <typename info_pta_t, typename info_t, bool nobag>
    class SplitEmbeddingBackwardCountUniqueIndicesKernel {
    public:
        SplitEmbeddingBackwardCountUniqueIndicesKernel(
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                _sorted_linear_indices_num_runs,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                _sorted_linear_indices_cumulative_run_lengths,
            const at::PackedTensorAccessor32<info_pta_t, 1, RestrictPtrTraits>
                _sorted_infos,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                _weights_placements,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits>
                _dev_or_uvm_unique_indices,
            const int _info_B_num_bits
        ) : sorted_linear_indices_num_runs(_sorted_linear_indices_num_runs),
            sorted_linear_indices_cumulative_run_lengths(_sorted_linear_indices_cumulative_run_lengths),
            sorted_infos(_sorted_infos),
            weights_placements(_weights_placements),
            dev_or_uvm_unique_indices(_dev_or_uvm_unique_indices),
            info_B_num_bits(_info_B_num_bits) {};

        void operator()(const sycl::nd_item<1>& item) const;
    
    private:
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths;
        const at::PackedTensorAccessor32<info_pta_t, 1, RestrictPtrTraits> sorted_infos;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> dev_or_uvm_unique_indices;
        const int info_B_num_bits;
    };

    template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    typename index_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    bool kUseVecBlocking>
    class SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelCTAPerRow {
    public:        
        SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelCTAPerRow(
            const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> _grad_output,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> _dev_weights,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> _uvm_weights,
            at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> _lxu_cache_weights,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _weights_placements, // if optimizer != "none"
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _weights_offsets,
            int64_t _D,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _hash_size_cumsum,
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> _sorted_linear_indices_run,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _sorted_linear_indices_cumulative_run_lengths,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _long_run_ids,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _num_long_run_ids,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _sorted_infos,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _sorted_lxu_cache_locations,
            const bool _use_uniq_cache_locations,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _table_unique_indices_offsets,
            bool _stochastic_rounding,
            PhiloxXpuState _stochastic_rounding_philox_args, // if not dense and optimizer != "none"
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _long_run_id_to_really_long_run_ids,
            at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits> _temp_grad_accum,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _grad_accum_counter,
            const int32_t _max_segment_length_per_cta,
            const bool _use_deterministic_algorithms,
            const int32_t _max_vecs_per_thread,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> _momentum1_dev,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> _momentum1_uvm,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _momentum1_placements,
            at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _momentum1_offsets,
            float _learning_rate = 0,
            float _eps = 0,
            float _weight_decay = 0.0,
            int64_t _weight_decay_mode = 0,
            float _max_norm = 0.0
        ) : grad_output(_grad_output),
            dev_weights(_dev_weights),
            uvm_weights(_uvm_weights),
            lxu_cache_weights(_lxu_cache_weights),
            weights_placements(_weights_placements),
            weights_offsets(_weights_offsets),
            D(_D),
            hash_size_cumsum(_hash_size_cumsum),
            sorted_linear_indices_run(_sorted_linear_indices_run),
            sorted_linear_indices_cumulative_run_lengths(_sorted_linear_indices_cumulative_run_lengths),
            long_run_ids(_long_run_ids),
            num_long_run_ids(_num_long_run_ids),
            sorted_infos(_sorted_infos),
            sorted_lxu_cache_locations(_sorted_lxu_cache_locations),
            use_uniq_cache_locations(_use_uniq_cache_locations),
            table_unique_indices_offsets(_table_unique_indices_offsets),
            stochastic_rounding(_stochastic_rounding),
            stochastic_rounding_philox_args(_stochastic_rounding_philox_args),
            long_run_id_to_really_long_run_ids(_long_run_id_to_really_long_run_ids),
            temp_grad_accum(_temp_grad_accum),
            grad_accum_counter(_grad_accum_counter),
            max_segment_length_per_cta(_max_segment_length_per_cta),
            use_deterministic_algorithms(_use_deterministic_algorithms),
            max_vecs_per_thread(_max_vecs_per_thread),
            momentum1_dev(_momentum1_dev),
            momentum1_uvm(_momentum1_uvm),
            momentum1_placements(_momentum1_placements),
            momentum1_offsets(_momentum1_offsets),
            learning_rate(_learning_rate),
            eps(_eps),
            weight_decay(_weight_decay),
            weight_decay_mode(_weight_decay_mode),
            max_norm(_max_norm) {};

            void operator()(const sycl::nd_item<2>& item) const;

        private:
            const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output;
            mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights;
            mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights;
            mutable at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets;
            int64_t D;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum;
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_ids;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> num_long_run_ids;
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations;
            const bool use_uniq_cache_locations;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets;
            bool stochastic_rounding;
            PhiloxXpuState stochastic_rounding_philox_args;
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> long_run_id_to_really_long_run_ids;
            mutable at::PackedTensorAccessor32<at::acc_type<cache_t, true>, 2, RestrictPtrTraits> temp_grad_accum;
            mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> grad_accum_counter;
            const int32_t max_segment_length_per_cta;
            const bool use_deterministic_algorithms;
            const int32_t max_vecs_per_thread;
            mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev;
            mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm;
            mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements;
            mutable at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets;
            float learning_rate;
            float eps;
            float weight_decay;
            int64_t weight_decay_mode;
            float max_norm;
    };

    template <
    typename emb_t,
    typename grad_t,
    typename cache_t,
    typename index_t,
    int32_t kFixedMaxVecsPerThread,
    int32_t kThreadGroupSize,
    bool kUseVecBlocking>
    class SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelWarpPerRow {
    public:
        SplitEmbeddingNobagBackwardCodegenRowwiseAdagradUnweightedKernelWarpPerRow(
            const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> _grad_output,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> _dev_weights,
            at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> _uvm_weights,
            at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> _lxu_cache_weights,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _weights_placements,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _weights_offsets,
            int64_t _D,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _hash_size_cumsum,
            const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> _sorted_linear_indices_run,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _sorted_linear_indices_cumulative_run_lengths,
            const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _sorted_infos,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _sorted_lxu_cache_locations,
            const bool _use_uniq_cache_locations,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _table_unique_indices_offsets,
            const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _sorted_linear_indices_num_runs,
            int32_t _max_segment_length_per_warp,
            bool _stochastic_rounding,
            PhiloxXpuState _stochastic_rounding_philox_args, // if not dense and optimizer != "none"
            const int32_t _max_D,
            const int32_t _max_vecs_per_thread,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> _momentum1_dev,
            at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> _momentum1_uvm,
            at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> _momentum1_placements,
            at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> _momentum1_offsets,
            float _learning_rate = 0,
            float _eps = 0,
            float _weight_decay = 0.0,
            int64_t _weight_decay_mode = 0,
            float _max_norm = 0.0
        ):  grad_output(_grad_output),
            dev_weights(_dev_weights),
            uvm_weights(_uvm_weights),
            lxu_cache_weights(_lxu_cache_weights),
            weights_placements(_weights_placements),
            weights_offsets(_weights_offsets),
            D(_D),
            hash_size_cumsum(_hash_size_cumsum),
            sorted_linear_indices_run(_sorted_linear_indices_run),
            sorted_linear_indices_cumulative_run_lengths(_sorted_linear_indices_cumulative_run_lengths),
            sorted_infos(_sorted_infos),
            sorted_lxu_cache_locations(_sorted_lxu_cache_locations),
            use_uniq_cache_locations(_use_uniq_cache_locations),
            table_unique_indices_offsets(_table_unique_indices_offsets),
            sorted_linear_indices_num_runs(_sorted_linear_indices_num_runs),
            max_segment_length_per_warp(_max_segment_length_per_warp),
            stochastic_rounding(_stochastic_rounding),
            stochastic_rounding_philox_args(_stochastic_rounding_philox_args),
            max_D(_max_D),
            max_vecs_per_thread(_max_vecs_per_thread),
            momentum1_dev(_momentum1_dev),
            momentum1_uvm(_momentum1_uvm),
            momentum1_placements(_momentum1_placements),
            momentum1_offsets(_momentum1_offsets),
            learning_rate(_learning_rate),
            eps(_eps),
            weight_decay(_weight_decay),
            weight_decay_mode(_weight_decay_mode),
            max_norm(_max_norm) {};

        void operator()(const sycl::nd_item<2>& item) const;

    private:
        const at::PackedTensorAccessor64<grad_t, 2, RestrictPtrTraits> grad_output;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> dev_weights;
        mutable at::PackedTensorAccessor64<emb_t, 1, RestrictPtrTraits> uvm_weights;
        mutable at::PackedTensorAccessor64<cache_t, 2, RestrictPtrTraits> lxu_cache_weights;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> weights_placements;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> weights_offsets;
        int64_t D;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> hash_size_cumsum;
        const at::PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> sorted_linear_indices_run;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_cumulative_run_lengths;
        const at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> sorted_infos;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_lxu_cache_locations;
        const bool use_uniq_cache_locations;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> table_unique_indices_offsets;
        const at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> sorted_linear_indices_num_runs;
        int32_t max_segment_length_per_warp;
        bool stochastic_rounding;
        PhiloxXpuState stochastic_rounding_philox_args;
        const int32_t max_D;
        const int32_t max_vecs_per_thread;
        mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_dev;
        mutable at::PackedTensorAccessor64<at::acc_type<cache_t, true>, 1, RestrictPtrTraits> momentum1_uvm;
        mutable at::PackedTensorAccessor32<int32_t, 1, RestrictPtrTraits> momentum1_placements;
        mutable at::PackedTensorAccessor32<int64_t, 1, RestrictPtrTraits> momentum1_offsets;
        float learning_rate;
        float eps;
        float weight_decay;
        int64_t weight_decay_mode;
        float max_norm;
    };
} // namespace at::native::xpu
