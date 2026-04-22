# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0


# BSD License
#
# For FBGEMM software
#
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Owner(s): ["module: intel"]

import copy
import enum
import math
import sys
import unittest
from types import ModuleType
from typing import List, Optional

import torch

# Define fbgemm ops schemas here since we cannot register them in torch-xpu-ops.
# otherwise, it will fail fbgemm lib due to duplicate schema registration.
# for user, they can import fbgemm_gpu first before accessing fbgemm ops on xpu.
try:
    lib = torch.library.Library("fbgemm", "DEF")  # noqa: TOR901
except RuntimeError as err:
    # Pytest can import multiple test files in one process.
    # Re-open the namespace as a fragment if it was already defined.
    if "Only a single TORCH_LIBRARY can be used" not in str(err):
        raise
    lib = torch.library.Library("fbgemm", "FRAGMENT")  # noqa: TOR901


def _safe_define(schema: str) -> None:
    try:
        lib.define(schema)
    except RuntimeError as err:
        err_msg = str(err)
        if "multiple times" in err_msg or "already" in err_msg:
            return
        raise


_safe_define("split_embedding_codegen_lookup_rowwise_adagrad_function_pt2("
                "    Tensor placeholder_autograd_tensor, "
                "    Tensor[](a!) weights, "
                "    Tensor D_offsets, "
                "    SymInt total_D, "
                "    SymInt max_D, "
                "    Tensor hash_size_cumsum, "
                "    int total_hash_size_bits, "
                "    Tensor indices, "
                "    Tensor offsets, "
                "    int pooling_mode, "
                "    Tensor? indice_weights, "
                "    Tensor? feature_requires_grad, "
                "    int output_dtype, "
                "    Tensor?[](e!) aux_tensor, "
                "    int[] aux_int, "
                "    float[] aux_float, "
                "    bool[] aux_bool, "
                "    Tensor[](g!) momentum1, "
                "    Tensor learning_rate_tensor, "
                "    int[] optim_int, "
                "    float[] optim_float, "
                "    SymInt max_B=-1, "
                "    SymInt max_B_feature_rank=-1, "
                "    SymInt vbe_output_size=-1 "
                ") -> Tensor")

_safe_define("split_embedding_nobag_codegen_forward_unweighted_pt2_wrapper("
                "    Tensor host_weights, "
                "    Tensor dev_weights, "
                "    Tensor uvm_weights, "
                "    Tensor lxu_cache_weights, "
                "    Tensor weights_placements, "
                "    Tensor weights_offsets, "
                "    SymInt D, "
                "    Tensor hash_size_cumsum, "
                "    Tensor indices, "
                "    Tensor offsets, "
                "    Tensor lxu_cache_locations, "
                "    Tensor(f!) uvm_cache_stats, "
                "    bool is_experimental, "
                "    int output_dtype "
                ") -> Tensor")

_safe_define("split_embedding_nobag_codegen_forward_unweighted_xpu("
                "    Tensor dev_weights, "
                "    Tensor uvm_weights, "
                "    Tensor lxu_cache_weights, "
                "    Tensor weights_placements, "
                "    Tensor weights_offsets, "
                "    SymInt D, "
                "    Tensor indices, "
                "    Tensor offsets, "
                "    Tensor lxu_cache_locations, "
                "    Tensor uvm_cache_stats, "
                "    int output_dtype, "
                "    bool is_experimental"
                ") -> Tensor")

_safe_define("split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_pt2_wrapper("
                "    Tensor grad_output, "
                "    Tensor(a!) host_weights, "
                "    Tensor(b!) dev_weights, "
                "    Tensor(c!) uvm_weights, "
                "    Tensor(d!) lxu_cache_weights, "
                "    Tensor weights_placements, "
                "    Tensor weights_offsets, "
                "    SymInt D, "
                "    Tensor hash_size_cumsum, "
                "    int total_hash_size_bits, "
                "    Tensor indices, "
                "    Tensor offsets, "
                "    Tensor lxu_cache_locations, "
                "    int BT_block_size, "
                "    int max_segment_length_per_warp, "
                "    bool stochastic_rounding, "
                "    int info_B_num_bits, "
                "    int info_B_mask_int64, "
                "    bool use_uniq_cache_locations, "
                "    bool use_homogeneous_placements,"
                "    Tensor(g!) momentum1_host, Tensor(h!) momentum1_dev, Tensor(i!) momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets, Tensor learning_rate_tensor, float eps = 0, float weight_decay = 0.0, int weight_decay_mode = 0, float max_norm = 0.0 "
                ") -> Tensor")

_safe_define("split_embedding_nobag_backward_codegen_rowwise_adagrad_unweighted_exact_xpu("
                "    Tensor grad_output, "
                "    Tensor(a!) dev_weights, "
                "    Tensor(b!) uvm_weights, "
                "    Tensor lxu_cache_weights, "
                "    Tensor weights_placements, "
                "    Tensor weights_offsets, "
                "    SymInt D, "
                "    Tensor hash_size_cumsum, "
                "    int total_hash_size_bits, "
                "    Tensor indices, "
                "    Tensor offsets, "
                "    Tensor lxu_cache_locations, "
                "    int unused_, "
                "    int max_segment_length_per_warp, "
                "    bool stochastic_rounding, "
                "    int info_B_num_bits, "
                "    int info_B_mask_int64, "
                "    bool use_uniq_cache_locations, "
                "    bool use_homogeneous_placements, "
                "    Tensor(h!) momentum1_dev, Tensor(i!) momentum1_uvm, Tensor momentum1_placements, Tensor momentum1_offsets, Tensor learning_rate_tensor, float eps = 0, float weight_decay = 0.0, int weight_decay_mode = 0, float max_norm = 0.0"
                ") -> Tensor")


class PoolingMode(enum.IntEnum):
    SUM = 0
    MEAN = 1
    NONE = 2


# PlacementType matching the C++ kernel enum
class PlacementType(enum.IntEnum):
    DEVICE = 0
    MANAGED = 1
    MANAGED_CACHING = 2


def _ensure_fbgemm_gpu_mock() -> None:
    if "fbgemm_gpu" in sys.modules:
        return
    fbgemm_gpu_mock = ModuleType("fbgemm_gpu")
    sys.modules["fbgemm_gpu"] = fbgemm_gpu_mock
    split_table_ops_mock = ModuleType("split_table_batched_embeddings_ops_common")
    split_table_ops_mock.PoolingMode = PoolingMode
    sys.modules["fbgemm_gpu.split_table_batched_embeddings_ops_common"] = split_table_ops_mock
    fbgemm_gpu_mock.split_table_batched_embeddings_ops_common = split_table_ops_mock


_ensure_fbgemm_gpu_mock()


class SparseType(enum.IntEnum):
    FP32 = 0
    FP16 = 1
    BF16 = 5


def _generate_uniq_cache_locations_data(
    T: int,
    B: int,
    D: int,
    num_rows_per_table: int,
    L: int,
    placements: List[int],
    learning_rate: float = 0.5,
    eps: float = 0.2,
    output_dtype: SparseType = SparseType.FP32,
    num_cache_slots: int = 64,
    info_B_num_bits: int = 29,
    use_homogeneous_placements: bool = False,
    seed: int = 42,
):
    """
    Generate data with use_uniq_cache_locations=True.

    Each row appears at most once per table and indices are pre-sorted so
    natural order matches sorted linear index order. Cache slots are unique
    (each slot used by at most one row). This exercises the
    lxu_cache_locations.size(0) > 0 branch with use_uniq_cache_locations=true
    in the backward kernel.

    L must be <= num_rows_per_table to guarantee uniqueness.
    """
    assert len(placements) == T
    assert L <= num_rows_per_table
    torch.manual_seed(seed)

    total_rows = T * num_rows_per_table
    total_D = T * D
    max_D = D

    weights_placements = torch.tensor(placements, dtype=torch.int32)

    dev_offset = 0
    uvm_offset = 0
    offsets_list = []
    for t in range(T):
        if placements[t] == PlacementType.DEVICE:
            offsets_list.append(dev_offset)
            dev_offset += num_rows_per_table * D
        else:
            offsets_list.append(uvm_offset)
            uvm_offset += num_rows_per_table * D

    weights_offsets = torch.tensor(offsets_list, dtype=torch.int64)
    dev_weights = torch.randn(dev_offset, dtype=torch.float32) if dev_offset > 0 else torch.zeros(0, dtype=torch.float32)
    uvm_weights = torch.randn(uvm_offset, dtype=torch.float32) if uvm_offset > 0 else torch.zeros(0, dtype=torch.float32)

    has_caching = any(p == PlacementType.MANAGED_CACHING for p in placements)
    if has_caching and num_cache_slots > 0:
        lxu_cache_weights = torch.randn(num_cache_slots, D, dtype=torch.float32)
    else:
        lxu_cache_weights = torch.zeros(0, 0, dtype=torch.float32)

    D_offsets = torch.tensor([i * D for i in range(T + 1)], dtype=torch.int32)
    hash_size_cumsum = torch.tensor(
        [i * num_rows_per_table for i in range(T + 1)], dtype=torch.int64
    )
    total_hash_size_bits = int(math.ceil(math.log2(max(total_rows, 1)))) + 1

    total_indices = T * B * L
    indices = torch.zeros(total_indices, dtype=torch.int64)
    for t in range(T):
        for b in range(B):
            perm = torch.randperm(num_rows_per_table)[:L]
            start = (t * B + b) * L
            indices[start:start + L] = perm

    offsets = torch.arange(0, total_indices + 1, L, dtype=torch.int64)
    assert offsets.numel() == T * B + 1

    # Pre-sort within each table section so natural order == sorted linear index order
    for t in range(T):
        start_idx = t * B * L
        end_idx = (t + 1) * B * L
        section = indices[start_idx:end_idx]
        sorted_section, _ = torch.sort(section)
        indices[start_idx:end_idx] = sorted_section

    if has_caching and num_cache_slots > 0:
        lxu_cache_locations = torch.full((total_indices,), -1, dtype=torch.int32)
        global_slot_idx = 0
        for t in range(T):
            if placements[t] != PlacementType.MANAGED_CACHING:
                continue
            table_offset = offsets_list[t]
            start_idx = t * B * L
            end_idx = (t + 1) * B * L
            unique_rows = indices[start_idx:end_idx].unique().sort()[0].tolist()

            row_to_slot = {}
            for row in unique_rows:
                if global_slot_idx >= num_cache_slots:
                    break
                row_to_slot[row] = global_slot_idx
                src_start = table_offset + row * D
                lxu_cache_weights[global_slot_idx] = uvm_weights[src_start:src_start + D]
                global_slot_idx += 1

            for i in range(start_idx, end_idx):
                row_idx = indices[i].item()
                if row_idx in row_to_slot:
                    lxu_cache_locations[i] = row_to_slot[row_idx]
    else:
        lxu_cache_locations = torch.zeros(0, dtype=torch.int32)

    momentum1_dev = torch.zeros(total_rows, dtype=torch.float32)
    momentum1_host = torch.zeros(0, dtype=torch.float32)
    momentum1_placements = torch.zeros(T, dtype=torch.int32)
    momentum1_offsets = torch.tensor(
        [i * num_rows_per_table for i in range(T)], dtype=torch.int64
    )

    weights = [dev_weights, uvm_weights, weights_placements, weights_offsets, lxu_cache_weights]
    momentum1 = [momentum1_dev, momentum1_host, momentum1_placements, momentum1_offsets]

    learning_rate_tensor = torch.tensor(learning_rate, dtype=torch.float32)
    info_B_mask = (1 << info_B_num_bits) - 1

    aux_tensor: List[Optional[torch.Tensor]] = [
        None, None, None, lxu_cache_locations, None, None, None
    ]
    aux_int = [0, info_B_num_bits, info_B_mask]
    aux_float = [0.0, 1.0]
    # Matches C++ ArgIndex_aux_bool enum in pt2_arg_utils.h
    aux_bool = [
        False,                       # IDX_IS_EXPERIMENTAL_TBE = 0
        True,                        # IDX_USE_UNIQ_CACHE_LOCATIONS_BWD = 1
        use_homogeneous_placements,  # IDX_USE_HOMOGENEOUS_PLACEMENTS = 2
        False,                       # IDX_APPLY_GLOBAL_WEIGHT_DECAY = 3
        False,                       # IDX_GRADIENT_CLIPPING = 4
        False,                       # IDX_STOCHASTIC_ROUNDING = 5
        False,                       # IDX_MIXED_D = 6
        False,                       # reserved
    ]

    optim_float = [eps, 0.0, 0.0]
    optim_int = [0]

    inputs = {
        "placeholder_autograd_tensor": torch.zeros(0, dtype=torch.float32),
        "weights": weights,
        "D_offsets": D_offsets,
        "total_D": total_D,
        "max_D": max_D,
        "hash_size_cumsum": hash_size_cumsum,
        "total_hash_size_bits": total_hash_size_bits,
        "indices": indices,
        "offsets": offsets,
        "pooling_mode": int(PoolingMode.NONE),
        "indice_weights": None,
        "feature_requires_grad": None,
        "output_dtype": int(output_dtype),
        "max_B": -1,
        "max_B_feature_rank": -1,
        "vbe_output_size": -1,
        "aux_tensor": aux_tensor,
        "aux_int": aux_int,
        "aux_float": aux_float,
        "aux_bool": aux_bool,
        "learning_rate_tensor": learning_rate_tensor,
        "momentum1": momentum1,
        "optim_int": optim_int,
        "optim_float": optim_float,
    }
    return inputs


def _generate_synthetic_data(
    T: int,
    B: int,
    D: int,
    num_rows_per_table: int,
    L: int,
    placements: List[int],
    learning_rate: float = 0.5,
    eps: float = 0.2,
    output_dtype: SparseType = SparseType.FP16,
    num_cache_slots: int = 0,
    cache_hit_ratio: float = 0.7,
    info_B_num_bits: int = 29,
    use_homogeneous_placements: bool = False,
    is_experimental: bool = False,
    seed: int = 42,
):
    """
    Generate synthetic input data for the split_embedding backward adagrad operator.

    Supports all placement types:
      - PlacementType.DEVICE (0): table stored in dev_weights buffer
      - PlacementType.MANAGED (1): table stored in uvm_weights buffer
      - PlacementType.MANAGED_CACHING (2): table stored in uvm_weights with LXU cache

    Args:
        T: Number of embedding tables
        B: Batch size
        D: Embedding dimension (per table)
        num_rows_per_table: Number of rows in each embedding table
        L: Number of indices per sample (bag size)
        placements: List of PlacementType values per table
        learning_rate: AdaGrad learning rate
        eps: AdaGrad epsilon
        output_dtype: Output precision
        num_cache_slots: Number of LXU cache slots (0 = no cache)
        cache_hit_ratio: Fraction of indices that hit the cache (for MANAGED_CACHING)
        info_B_num_bits: Bits for encoding batch index
        use_homogeneous_placements: Whether all tables share the same placement
        is_experimental: Use TBEv2 experimental path
        seed: Random seed for reproducibility
    """
    assert len(placements) == T
    torch.manual_seed(seed)

    total_rows = T * num_rows_per_table
    total_D = T * D
    max_D = D

    weights_placements = torch.tensor(placements, dtype=torch.int32)

    # Compute per-buffer offsets: DEVICE tables go into dev_weights,
    # MANAGED and MANAGED_CACHING tables go into uvm_weights
    dev_offset = 0
    uvm_offset = 0
    offsets_list = []
    for t in range(T):
        if placements[t] == PlacementType.DEVICE:
            offsets_list.append(dev_offset)
            dev_offset += num_rows_per_table * D
        else:
            offsets_list.append(uvm_offset)
            uvm_offset += num_rows_per_table * D

    weights_offsets = torch.tensor(offsets_list, dtype=torch.int64)

    dev_weights = torch.randn(dev_offset, dtype=torch.float32) if dev_offset > 0 else torch.zeros(0, dtype=torch.float32)
    uvm_weights = torch.randn(uvm_offset, dtype=torch.float32) if uvm_offset > 0 else torch.zeros(0, dtype=torch.float32)

    # LXU cache for MANAGED_CACHING tables
    has_caching = any(p == PlacementType.MANAGED_CACHING for p in placements)
    if has_caching and num_cache_slots > 0:
        lxu_cache_weights = torch.randn(num_cache_slots, D, dtype=torch.float32)
    else:
        lxu_cache_weights = torch.zeros(0, 0, dtype=torch.float32)

    D_offsets = torch.tensor([i * D for i in range(T + 1)], dtype=torch.int32)
    hash_size_cumsum = torch.tensor(
        [i * num_rows_per_table for i in range(T + 1)], dtype=torch.int64
    )
    total_hash_size_bits = int(math.ceil(math.log2(max(total_rows, 1)))) + 1

    # Generate indices and offsets for PoolingMode.NONE (nobag)
    total_indices = T * B * L
    indices = torch.randint(0, num_rows_per_table, (total_indices,), dtype=torch.int64)
    offsets = torch.arange(0, total_indices + 1, L, dtype=torch.int64)
    assert offsets.numel() == T * B + 1

    # Generate lxu_cache_locations.
    # In a real TBE, each (table, row) pair maps to exactly one cache slot.
    # We assign a globally unique cache slot per cached row across all tables.
    if has_caching and num_cache_slots > 0:
        lxu_cache_locations = torch.full((total_indices,), -1, dtype=torch.int32)
        global_slot_idx = 0
        for t in range(T):
            if placements[t] != PlacementType.MANAGED_CACHING:
                continue
            table_offset = offsets_list[t]
            # Decide how many rows of this table to cache
            slots_remaining = num_cache_slots - global_slot_idx
            max_cacheable = min(num_rows_per_table, slots_remaining)
            num_hits_rows = int(max_cacheable * cache_hit_ratio)
            cached_rows = torch.randperm(num_rows_per_table)[:num_hits_rows]
            row_to_slot = {}
            for row in cached_rows.tolist():
                if global_slot_idx >= num_cache_slots:
                    break
                row_to_slot[row] = global_slot_idx
                src_start = table_offset + row * D
                lxu_cache_weights[global_slot_idx] = uvm_weights[src_start:src_start + D]
                global_slot_idx += 1

            # Fill lxu_cache_locations for this table's indices
            start = t * B * L
            end = (t + 1) * B * L
            for i in range(start, end):
                row_idx = indices[i].item()
                if row_idx in row_to_slot:
                    lxu_cache_locations[i] = row_to_slot[row_idx]
    else:
        lxu_cache_locations = torch.zeros(0, dtype=torch.int32)

    # Momentum (one per row, all placements=0 per dataset convention)
    momentum1_dev = torch.zeros(total_rows, dtype=torch.float32)
    momentum1_host = torch.zeros(0, dtype=torch.float32)
    momentum1_placements = torch.zeros(T, dtype=torch.int32)
    momentum1_offsets = torch.tensor(
        [i * num_rows_per_table for i in range(T)], dtype=torch.int64
    )

    weights = [
        dev_weights,
        uvm_weights,
        weights_placements,
        weights_offsets,
        lxu_cache_weights,
    ]
    momentum1 = [
        momentum1_dev,
        momentum1_host,
        momentum1_placements,
        momentum1_offsets,
    ]

    learning_rate_tensor = torch.tensor(learning_rate, dtype=torch.float32)

    info_B_mask = (1 << info_B_num_bits) - 1
    aux_tensor: List[Optional[torch.Tensor]] = [
        None, None, None, lxu_cache_locations, None, None, None
    ]
    aux_int = [0, info_B_num_bits, info_B_mask]
    aux_float = [0.0, 1.0]
    aux_bool = [
        False,                       # IDX_GRADIENT_CLIPPING
        False,                       # IDX_STOCHASTIC_ROUNDING
        is_experimental,             # IDX_IS_EXPERIMENTAL_TBE
        False,                       # IDX_USE_UNIQ_CACHE_LOCATIONS_BWD
        False,                       # IDX_APPLY_GLOBAL_WEIGHT_DECAY
        use_homogeneous_placements,  # IDX_USE_HOMOGENEOUS_PLACEMENTS
        False,                       # reserved
    ]

    optim_float = [eps, 0.0, 0.0]
    optim_int = [0]

    inputs = {
        "placeholder_autograd_tensor": torch.zeros(0, dtype=torch.float32),
        "weights": weights,
        "D_offsets": D_offsets,
        "total_D": total_D,
        "max_D": max_D,
        "hash_size_cumsum": hash_size_cumsum,
        "total_hash_size_bits": total_hash_size_bits,
        "indices": indices,
        "offsets": offsets,
        "pooling_mode": int(PoolingMode.NONE),
        "indice_weights": None,
        "feature_requires_grad": None,
        "output_dtype": int(output_dtype),
        "max_B": -1,
        "max_B_feature_rank": -1,
        "vbe_output_size": -1,
        "aux_tensor": aux_tensor,
        "aux_int": aux_int,
        "aux_float": aux_float,
        "aux_bool": aux_bool,
        "learning_rate_tensor": learning_rate_tensor,
        "momentum1": momentum1,
        "optim_int": optim_int,
        "optim_float": optim_float,
    }
    return inputs


def _get_flattened_weights(inputs):
    """
    Reconstruct a single flattened weight tensor from dev + uvm buffers
    ordered by table, for use in the reference implementation.
    """
    weights = inputs["weights"]
    dev_weights = weights[0]
    uvm_weights = weights[1]
    placements = weights[2].tolist()
    offsets = weights[3].tolist()
    hash_size_cumsum = inputs["hash_size_cumsum"]
    D_offsets = inputs["D_offsets"]
    T = len(placements)

    total_elements = 0
    for t in range(T):
        num_rows = (hash_size_cumsum[t + 1] - hash_size_cumsum[t]).item()
        D = (D_offsets[t + 1] - D_offsets[t]).item()
        total_elements += num_rows * D

    flat = torch.zeros(total_elements, dtype=torch.float32)
    flat_offset = 0
    for t in range(T):
        num_rows = (hash_size_cumsum[t + 1] - hash_size_cumsum[t]).item()
        D = (D_offsets[t + 1] - D_offsets[t]).item()
        n_elems = num_rows * D
        if n_elems == 0:
            continue
        buf_offset = offsets[t]
        if placements[t] == PlacementType.DEVICE:
            flat[flat_offset:flat_offset + n_elems] = dev_weights[buf_offset:buf_offset + n_elems]
        else:
            flat[flat_offset:flat_offset + n_elems] = uvm_weights[buf_offset:buf_offset + n_elems]
        flat_offset += n_elems
    return flat


def _reference_adagrad_update(
    flat_weights: torch.Tensor,
    momentum1: torch.Tensor,
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    hash_size_cumsum: torch.Tensor,
    D_offsets: torch.Tensor,
    learning_rate: float,
    eps: float,
):
    """
    Reference CPU implementation of nobag rowwise AdaGrad.

    The kernel first accumulates gradients for all occurrences of the same row
    (across the entire batch), then applies a single AdaGrad update using
    momentum = mean(accumulated_grad^2) per row.
    """
    T = hash_size_cumsum.numel() - 1
    total_B = offsets.numel() - 1
    B = total_B // T

    updated_weights = flat_weights.clone().float()
    updated_momentum = momentum1.clone().float()
    grad_output_f32 = grad_output.float()

    for t in range(T):
        D = (D_offsets[t + 1] - D_offsets[t]).item()
        hash_offset = hash_size_cumsum[t].item()
        weight_offset = hash_offset * D

        row_grads: dict = {}
        for b in range(B):
            bag_idx = t * B + b
            idx_start = offsets[bag_idx].item()
            idx_end = offsets[bag_idx + 1].item()

            for l_idx in range(idx_start, idx_end):
                row_idx = indices[l_idx].item()
                grad = grad_output_f32[l_idx, :D]
                if row_idx not in row_grads:
                    row_grads[row_idx] = torch.zeros(D)
                row_grads[row_idx] += grad

        for row_idx, accumulated_grad in row_grads.items():
            global_row = hash_offset + row_idx
            grad_sq_mean = (accumulated_grad * accumulated_grad).sum().item() / D
            updated_momentum[global_row] += grad_sq_mean

            multiplier = learning_rate / (
                math.sqrt(updated_momentum[global_row]) + eps
            )
            w_start = weight_offset + row_idx * D
            w_end = w_start + D
            updated_weights[w_start:w_end] -= multiplier * accumulated_grad

    return updated_weights, updated_momentum


def _scatter_weights_back(inputs, flat_weights):
    """
    Scatter the reference flat weights back into per-buffer tensors for comparison.
    Returns (dev_weights, uvm_weights) tensors.
    """
    weights = inputs["weights"]
    placements = weights[2].tolist()
    offsets = weights[3].tolist()
    hash_size_cumsum = inputs["hash_size_cumsum"]
    D_offsets = inputs["D_offsets"]
    T = len(placements)

    dev_weights = weights[0].clone()
    uvm_weights = weights[1].clone()

    flat_offset = 0
    for t in range(T):
        num_rows = (hash_size_cumsum[t + 1] - hash_size_cumsum[t]).item()
        D = (D_offsets[t + 1] - D_offsets[t]).item()
        n_elems = num_rows * D
        if n_elems == 0:
            flat_offset += n_elems
            continue
        buf_offset = offsets[t]
        if placements[t] == PlacementType.DEVICE:
            dev_weights[buf_offset:buf_offset + n_elems] = flat_weights[flat_offset:flat_offset + n_elems]
        else:
            uvm_weights[buf_offset:buf_offset + n_elems] = flat_weights[flat_offset:flat_offset + n_elems]
        flat_offset += n_elems
    return dev_weights, uvm_weights


def _move_to_device(inputs, device):
    result = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.to(device)
        elif isinstance(v, list):
            result[k] = [
                x.to(device) if isinstance(x, torch.Tensor) else x for x in v
            ]
        else:
            result[k] = v
    return result


class TestSplitLookupBackwardSynthetic(unittest.TestCase):
    """
    Tests for split_embedding_codegen_lookup_rowwise_adagrad backward pass
    using synthetically generated data with various placement configurations.
    """

    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise unittest.SkipTest("XPU is not available")

    def setUp(self):
        torch.xpu.synchronize()

    def tearDown(self):
        torch.xpu.empty_cache()
        torch.xpu.synchronize()

    def _run_forward_backward(self, inputs_xpu, grad_output_xpu):
        weights = inputs_xpu["weights"]
        weights[0].requires_grad_(True)
        if weights[1].numel() > 0:
            weights[1].requires_grad_(True)

        result = torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_pt2(
            placeholder_autograd_tensor=inputs_xpu["placeholder_autograd_tensor"],
            weights=weights,
            D_offsets=inputs_xpu["D_offsets"],
            total_D=inputs_xpu["total_D"],
            max_D=inputs_xpu["max_D"],
            hash_size_cumsum=inputs_xpu["hash_size_cumsum"],
            total_hash_size_bits=inputs_xpu["total_hash_size_bits"],
            indices=inputs_xpu["indices"],
            offsets=inputs_xpu["offsets"],
            pooling_mode=inputs_xpu["pooling_mode"],
            indice_weights=inputs_xpu["indice_weights"],
            feature_requires_grad=inputs_xpu["feature_requires_grad"],
            output_dtype=inputs_xpu["output_dtype"],
            max_B=inputs_xpu["max_B"],
            max_B_feature_rank=inputs_xpu["max_B_feature_rank"],
            vbe_output_size=inputs_xpu["vbe_output_size"],
            aux_tensor=inputs_xpu["aux_tensor"],
            aux_int=inputs_xpu["aux_int"],
            aux_float=inputs_xpu["aux_float"],
            aux_bool=inputs_xpu["aux_bool"],
            learning_rate_tensor=inputs_xpu["learning_rate_tensor"].cpu(),
            momentum1=inputs_xpu["momentum1"],
            optim_int=inputs_xpu["optim_int"],
            optim_float=inputs_xpu["optim_float"],
        )

        result.backward(grad_output_xpu)
        torch.xpu.synchronize()
        return result

    def _verify_backward(self, inputs, atol, rtol):
        """
        Run forward+backward on XPU and compare weight updates against
        a reference CPU AdaGrad implementation.
        """
        total_indices = inputs["indices"].numel()
        D = inputs["max_D"]
        output_dtype_enum = SparseType(inputs["output_dtype"])
        if output_dtype_enum == SparseType.FP32:
            grad_dtype = torch.float32
        elif output_dtype_enum == SparseType.FP16:
            grad_dtype = torch.float16
        else:
            grad_dtype = torch.bfloat16

        torch.manual_seed(1234)
        grad_output = torch.randn(total_indices, D, dtype=grad_dtype)

        # Compute reference on CPU using flattened weight view
        flat_weights = _get_flattened_weights(inputs)
        ref_weights, ref_momentum = _reference_adagrad_update(
            flat_weights=flat_weights,
            momentum1=inputs["momentum1"][0].clone(),
            grad_output=grad_output,
            indices=inputs["indices"],
            offsets=inputs["offsets"],
            hash_size_cumsum=inputs["hash_size_cumsum"],
            D_offsets=inputs["D_offsets"],
            learning_rate=inputs["learning_rate_tensor"].item(),
            eps=inputs["optim_float"][0],
        )
        ref_dev, ref_uvm = _scatter_weights_back(inputs, ref_weights)

        # Run on XPU
        inputs_xpu = _move_to_device(copy.deepcopy(inputs), "xpu")
        grad_xpu = grad_output.to("xpu")
        self._run_forward_backward(inputs_xpu, grad_xpu)

        # Compare dev_weights (DEVICE placement tables)
        actual_dev = inputs_xpu["weights"][0].cpu().float()
        if actual_dev.numel() > 0:
            torch.testing.assert_close(
                actual_dev, ref_dev, atol=atol, rtol=rtol,
                msg="dev_weights update mismatch vs reference AdaGrad",
            )

        # Compare uvm_weights and lxu_cache_weights for MANAGED_CACHING tables.
        # The kernel writes updated weights to lxu_cache_weights[slot] for cache
        # hits and to uvm_weights directly for cache misses.
        # Reconstruct the effective weights per-row and compare against reference.
        placements = inputs["weights"][2].tolist()
        has_caching = any(p == PlacementType.MANAGED_CACHING for p in placements)

        if has_caching:
            self._verify_caching_weights(
                inputs, inputs_xpu, ref_uvm, atol, rtol
            )
        else:
            actual_uvm = inputs_xpu["weights"][1].cpu().float()
            if actual_uvm.numel() > 0:
                torch.testing.assert_close(
                    actual_uvm, ref_uvm, atol=atol, rtol=rtol,
                    msg="uvm_weights update mismatch vs reference AdaGrad",
                )

        # Compare momentum
        actual_momentum = inputs_xpu["momentum1"][0].cpu().float()
        torch.testing.assert_close(
            actual_momentum, ref_momentum, atol=atol, rtol=rtol,
            msg="Momentum update mismatch vs reference AdaGrad",
        )

    def _verify_caching_weights(self, inputs, inputs_xpu, ref_uvm, atol, rtol):
        """
        For MANAGED_CACHING tables, the kernel writes to lxu_cache_weights[slot]
        for cache hits and uvm_weights for cache misses. Reconstruct the effective
        per-row updated weights and compare against reference.
        """
        placements = inputs["weights"][2].tolist()
        offsets_list = inputs["weights"][3].tolist()
        hash_size_cumsum = inputs["hash_size_cumsum"]
        D_offsets = inputs["D_offsets"]
        T = len(placements)

        actual_uvm = inputs_xpu["weights"][1].cpu().float()
        actual_cache = inputs_xpu["weights"][4].cpu().float()
        lxu_cache_locations = inputs["aux_tensor"][3]
        indices = inputs["indices"]
        input_offsets = inputs["offsets"]
        B = (input_offsets.numel() - 1) // T

        for t in range(T):
            if placements[t] != PlacementType.MANAGED_CACHING:
                continue

            D = (D_offsets[t + 1] - D_offsets[t]).item()
            num_rows = (hash_size_cumsum[t + 1] - hash_size_cumsum[t]).item()
            buf_offset = offsets_list[t]

            # For each row in this table, find the effective updated weight:
            # Check if any index pointing to this row had a cache hit
            for row in range(num_rows):
                ref_start = buf_offset + row * D
                ref_row = ref_uvm[ref_start:ref_start + D]

                # Find the last cache location used for this row
                cache_slot = -1
                table_start_idx = t * B
                for b in range(B):
                    bag_idx = table_start_idx + b
                    idx_start = input_offsets[bag_idx].item()
                    idx_end = input_offsets[bag_idx + 1].item()
                    for l_idx in range(idx_start, idx_end):
                        if indices[l_idx].item() == row:
                            loc = lxu_cache_locations[l_idx].item()
                            if loc >= 0:
                                cache_slot = loc

                if cache_slot >= 0:
                    actual_row = actual_cache[cache_slot, :D]
                else:
                    actual_row = actual_uvm[ref_start:ref_start + D]

                torch.testing.assert_close(
                    actual_row, ref_row, atol=atol, rtol=rtol,
                    msg=f"MANAGED_CACHING table {t} row {row} mismatch",
                )

    # ===================================================================
    # Placement: all DEVICE (0) — data in dev_weights only
    # ===================================================================

    def test_backward_all_device_fp32_small(self):
        """All tables on DEVICE, FP32, small config."""
        inputs = _generate_synthetic_data(
            T=2, B=4, D=8, num_rows_per_table=5, L=3,
            placements=[0, 0],
            output_dtype=SparseType.FP32, seed=100,
        )
        self._verify_backward(inputs, atol=1e-4, rtol=1e-4)

    def test_backward_all_device_fp16_small(self):
        """All tables on DEVICE, FP16, small config."""
        inputs = _generate_synthetic_data(
            T=2, B=4, D=8, num_rows_per_table=5, L=3,
            placements=[0, 0],
            output_dtype=SparseType.FP16, seed=101,
        )
        self._verify_backward(inputs, atol=1e-2, rtol=1e-2)

    def test_backward_all_device_fp32_medium(self):
        """All tables on DEVICE, FP32, medium config."""
        inputs = _generate_synthetic_data(
            T=5, B=32, D=64, num_rows_per_table=10, L=18,
            placements=[0, 0, 0, 0, 0],
            output_dtype=SparseType.FP32, seed=102,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_all_device_bf16_medium(self):
        """All tables on DEVICE, BF16 output, medium config."""
        inputs = _generate_synthetic_data(
            T=4, B=47, D=200, num_rows_per_table=10, L=18,
            placements=[0, 0, 0, 0],
            output_dtype=SparseType.BF16, seed=103,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    def test_backward_all_device_high_contention(self):
        """Few rows, many lookups — high contention on DEVICE."""
        inputs = _generate_synthetic_data(
            T=2, B=64, D=32, num_rows_per_table=3, L=10,
            placements=[0, 0],
            output_dtype=SparseType.FP32, seed=104,
        )
        self._verify_backward(inputs, atol=1e-2, rtol=1e-2)

    def test_backward_all_device_single_table(self):
        """Single table on DEVICE."""
        inputs = _generate_synthetic_data(
            T=1, B=16, D=128, num_rows_per_table=20, L=5,
            placements=[0],
            output_dtype=SparseType.FP32, seed=105,
        )
        self._verify_backward(inputs, atol=1e-4, rtol=1e-4)

    def test_backward_all_device_large_D(self):
        """Large embedding dimension on DEVICE."""
        inputs = _generate_synthetic_data(
            T=3, B=16, D=256, num_rows_per_table=10, L=8,
            placements=[0, 0, 0],
            output_dtype=SparseType.FP32, seed=106,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_all_device_info_B_30(self):
        """DEVICE placement with info_B_num_bits=30."""
        inputs = _generate_synthetic_data(
            T=2, B=82, D=8, num_rows_per_table=10, L=3,
            placements=[0, 0],
            info_B_num_bits=30,
            is_experimental=True,
            output_dtype=SparseType.FP32, seed=107,
        )
        self._verify_backward(inputs, atol=1e-4, rtol=1e-4)

    # ===================================================================
    # Placement: mixed DEVICE (0) + MANAGED (1) — data split across buffers
    # ===================================================================

    def test_backward_mixed_device_managed_fp32(self):
        """Mixed DEVICE/MANAGED placements, FP32."""
        inputs = _generate_synthetic_data(
            T=5, B=86, D=228, num_rows_per_table=10, L=18,
            placements=[0, 1, 1, 1, 0],
            output_dtype=SparseType.FP32, seed=200,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_mixed_device_managed_fp16(self):
        """Mixed DEVICE/MANAGED placements, FP16."""
        inputs = _generate_synthetic_data(
            T=5, B=86, D=228, num_rows_per_table=10, L=18,
            placements=[0, 1, 1, 1, 0],
            output_dtype=SparseType.FP16, seed=201,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    def test_backward_mixed_device_managed_bf16(self):
        """Mixed DEVICE/MANAGED placements, BF16."""
        inputs = _generate_synthetic_data(
            T=4, B=47, D=200, num_rows_per_table=10, L=18,
            placements=[0, 1, 1, 1],
            output_dtype=SparseType.BF16, seed=202,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    def test_backward_mixed_device_managed_6_tables(self):
        """6 tables with mixed placements."""
        inputs = _generate_synthetic_data(
            T=6, B=30, D=128, num_rows_per_table=10, L=8,
            placements=[0, 1, 1, 1, 0, 1],
            output_dtype=SparseType.FP16, seed=203,
            use_homogeneous_placements=False,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    # ===================================================================
    # Placement: all MANAGED_CACHING (2) — data in uvm + LXU cache
    # ===================================================================

    def test_backward_all_managed_caching_fp32(self):
        """All MANAGED_CACHING, FP32, with cache hits and misses."""
        inputs = _generate_synthetic_data(
            T=4, B=62, D=64, num_rows_per_table=10, L=13,
            placements=[2, 2, 2, 2],
            output_dtype=SparseType.FP32, seed=300,
            num_cache_slots=32,
            cache_hit_ratio=0.7,
            use_homogeneous_placements=True,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_all_managed_caching_fp16(self):
        """All MANAGED_CACHING, FP16."""
        inputs = _generate_synthetic_data(
            T=6, B=61, D=128, num_rows_per_table=10, L=18,
            placements=[2, 2, 2, 2, 2, 2],
            output_dtype=SparseType.FP16, seed=301,
            num_cache_slots=32,
            cache_hit_ratio=0.7,
            use_homogeneous_placements=True,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    def test_backward_all_managed_caching_bf16(self):
        """All MANAGED_CACHING, BF16."""
        inputs = _generate_synthetic_data(
            T=4, B=62, D=376, num_rows_per_table=10, L=13,
            placements=[2, 2, 2, 2],
            output_dtype=SparseType.BF16, seed=302,
            num_cache_slots=32,
            cache_hit_ratio=0.5,
            use_homogeneous_placements=True,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    def test_backward_all_managed_caching_full_hits(self):
        """All MANAGED_CACHING, 100% cache hit ratio."""
        inputs = _generate_synthetic_data(
            T=4, B=62, D=64, num_rows_per_table=10, L=13,
            placements=[2, 2, 2, 2],
            output_dtype=SparseType.FP32, seed=303,
            num_cache_slots=64,
            cache_hit_ratio=1.0,
            use_homogeneous_placements=True,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_mixed_placements_full_cache_hits(self):
        """Mixed DEVICE + MANAGED + MANAGED_CACHING, 100% cache hit ratio."""
        inputs = _generate_synthetic_data(
            T=4, B=62, D=64, num_rows_per_table=10, L=13,
            placements=[0, 1, 2, 1],
            output_dtype=SparseType.FP32, seed=303,
            num_cache_slots=64,
            cache_hit_ratio=1.0,
            use_homogeneous_placements=False,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_mixed_placements_full_cache_hits_fp16(self):
        """Mixed placements, 100% cache hit ratio, FP16."""
        inputs = _generate_synthetic_data(
            T=4, B=32, D=128, num_rows_per_table=10, L=8,
            placements=[0, 2, 1, 2],
            output_dtype=SparseType.FP16, seed=310,
            num_cache_slots=64,
            cache_hit_ratio=1.0,
            use_homogeneous_placements=False,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    def test_backward_mixed_placements_full_cache_hits_bf16(self):
        """Mixed placements, 100% cache hit ratio, BF16."""
        inputs = _generate_synthetic_data(
            T=4, B=32, D=128, num_rows_per_table=10, L=8,
            placements=[2, 0, 2, 1],
            output_dtype=SparseType.BF16, seed=311,
            num_cache_slots=64,
            cache_hit_ratio=1.0,
            use_homogeneous_placements=False,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    def test_backward_mixed_placements_full_cache_hits_large_D(self):
        """Mixed placements, 100% cache hit ratio, large embedding dimension."""
        inputs = _generate_synthetic_data(
            T=3, B=16, D=512, num_rows_per_table=10, L=6,
            placements=[2, 1, 0],
            output_dtype=SparseType.FP32, seed=312,
            num_cache_slots=64,
            cache_hit_ratio=1.0,
            use_homogeneous_placements=False,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_mixed_placements_full_cache_hits_high_lr(self):
        """Mixed placements, 100% cache hit ratio, high learning rate."""
        inputs = _generate_synthetic_data(
            T=4, B=16, D=64, num_rows_per_table=10, L=10,
            placements=[1, 2, 0, 2],
            learning_rate=2.0, eps=0.5,
            output_dtype=SparseType.FP32, seed=313,
            num_cache_slots=64,
            cache_hit_ratio=1.0,
            use_homogeneous_placements=False,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_mixed_placements_full_cache_hits_large_batch(self):
        """Mixed placements, 100% cache hit ratio, large batch size."""
        inputs = _generate_synthetic_data(
            T=3, B=128, D=64, num_rows_per_table=10, L=4,
            placements=[0, 2, 2],
            output_dtype=SparseType.FP32, seed=314,
            num_cache_slots=64,
            cache_hit_ratio=1.0,
            use_homogeneous_placements=False,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_all_managed_caching_all_misses(self):
        """All MANAGED_CACHING, 0% cache hit ratio (all misses, fallback to uvm)."""
        inputs = _generate_synthetic_data(
            T=4, B=62, D=64, num_rows_per_table=10, L=13,
            placements=[2, 2, 2, 2],
            output_dtype=SparseType.FP32, seed=304,
            num_cache_slots=32,
            cache_hit_ratio=0.0,
            use_homogeneous_placements=True,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_all_managed_caching_large_cache(self):
        """All MANAGED_CACHING with large cache (1024 slots)."""
        inputs = _generate_synthetic_data(
            T=6, B=61, D=428, num_rows_per_table=10, L=18,
            placements=[2, 2, 2, 2, 2, 2],
            output_dtype=SparseType.BF16, seed=305,
            num_cache_slots=1024,
            cache_hit_ratio=0.7,
            use_homogeneous_placements=True,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    # ===================================================================
    # Placement: mixed MANAGED_CACHING (2) + DEVICE (0)
    # ===================================================================

    def test_backward_mixed_caching_device_fp32(self):
        """Mixed MANAGED_CACHING + DEVICE, FP32."""
        inputs = _generate_synthetic_data(
            T=4, B=10, D=256, num_rows_per_table=10, L=100,
            placements=[2, 0, 0, 0],
            output_dtype=SparseType.FP32, seed=400,
            num_cache_slots=32,
            cache_hit_ratio=0.75,
            use_homogeneous_placements=False,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_backward_mixed_caching_device_fp16(self):
        """Mixed MANAGED_CACHING + DEVICE, FP16."""
        inputs = _generate_synthetic_data(
            T=3, B=10, D=256, num_rows_per_table=10, L=30,
            placements=[2, 0, 0],
            output_dtype=SparseType.FP16, seed=401,
            num_cache_slots=32,
            cache_hit_ratio=0.7,
            use_homogeneous_placements=False,
            info_B_num_bits=30,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    def test_backward_mixed_caching_device_with_cache_misses(self):
        """Mixed placements, partial cache misses (lxu_cache_locations has -1 values)."""
        inputs = _generate_synthetic_data(
            T=3, B=10, D=256, num_rows_per_table=10, L=35,
            placements=[2, 0, 0],
            output_dtype=SparseType.FP32, seed=402,
            num_cache_slots=32,
            cache_hit_ratio=0.25,
            use_homogeneous_placements=False,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    # ===================================================================
    # Varied optimizer / learning rate configs
    # ===================================================================

    def test_backward_varied_lr_low(self):
        """Low learning rate: lr=0.01."""
        inputs = _generate_synthetic_data(
            T=3, B=8, D=32, num_rows_per_table=10, L=6,
            placements=[0, 1, 0],
            learning_rate=0.01, eps=1e-8,
            output_dtype=SparseType.FP32, seed=500,
        )
        self._verify_backward(inputs, atol=1e-4, rtol=1e-4)

    def test_backward_varied_lr_high(self):
        """High learning rate: lr=2.0."""
        inputs = _generate_synthetic_data(
            T=3, B=8, D=32, num_rows_per_table=10, L=6,
            placements=[1, 1, 1],
            learning_rate=2.0, eps=0.5,
            output_dtype=SparseType.FP32, seed=501,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    # ===================================================================
    # Determinism tests
    # ===================================================================

    def test_determinism_device(self):
        """Determinism on DEVICE placement."""
        inputs = _generate_synthetic_data(
            T=3, B=16, D=64, num_rows_per_table=10, L=8,
            placements=[0, 0, 0],
            output_dtype=SparseType.FP32, seed=600,
        )
        self._verify_determinism(inputs)

    def test_determinism_mixed(self):
        """Determinism with mixed DEVICE + MANAGED."""
        inputs = _generate_synthetic_data(
            T=4, B=16, D=64, num_rows_per_table=10, L=8,
            placements=[0, 1, 1, 0],
            output_dtype=SparseType.FP32, seed=601,
        )
        self._verify_determinism(inputs)

    def test_determinism_caching(self):
        """Determinism with MANAGED_CACHING."""
        inputs = _generate_synthetic_data(
            T=3, B=10, D=64, num_rows_per_table=10, L=8,
            placements=[2, 0, 0],
            output_dtype=SparseType.FP32, seed=602,
            num_cache_slots=32,
            cache_hit_ratio=0.7,
        )
        self._verify_determinism(inputs)

    def _verify_determinism(self, inputs):
        total_indices = inputs["indices"].numel()
        D = inputs["max_D"]
        torch.manual_seed(700)
        grad_output = torch.randn(total_indices, D, dtype=torch.float32)

        results_dev = []
        results_uvm = []
        for _ in range(3):
            inputs_copy = copy.deepcopy(inputs)
            inputs_xpu = _move_to_device(inputs_copy, "xpu")
            grad_xpu = grad_output.to("xpu")
            self._run_forward_backward(inputs_xpu, grad_xpu)
            results_dev.append(inputs_xpu["weights"][0].cpu().clone())
            results_uvm.append(inputs_xpu["weights"][1].cpu().clone())

        for i in range(1, len(results_dev)):
            if results_dev[0].numel() > 0:
                torch.testing.assert_close(
                    results_dev[0], results_dev[i], atol=0, rtol=0,
                    msg=f"dev_weights: run {i} differs from run 0",
                )
            if results_uvm[0].numel() > 0:
                torch.testing.assert_close(
                    results_uvm[0], results_uvm[i], atol=0, rtol=0,
                    msg=f"uvm_weights: run {i} differs from run 0",
                )

    # ===================================================================
    # Forward output shape test
    # ===================================================================

    def test_forward_output_shape_device(self):
        """Forward output shape with DEVICE placement."""
        inputs = _generate_synthetic_data(
            T=3, B=8, D=64, num_rows_per_table=10, L=5,
            placements=[0, 0, 0],
            output_dtype=SparseType.FP32, seed=700,
        )
        self._verify_forward_shape(inputs)

    def test_forward_output_shape_mixed(self):
        """Forward output shape with mixed placements."""
        inputs = _generate_synthetic_data(
            T=4, B=8, D=64, num_rows_per_table=10, L=5,
            placements=[0, 1, 1, 0],
            output_dtype=SparseType.FP16, seed=701,
        )
        self._verify_forward_shape(inputs)

    def test_forward_output_shape_caching(self):
        """Forward output shape with MANAGED_CACHING."""
        inputs = _generate_synthetic_data(
            T=3, B=8, D=64, num_rows_per_table=10, L=5,
            placements=[2, 0, 0],
            output_dtype=SparseType.FP32, seed=702,
            num_cache_slots=32,
            cache_hit_ratio=0.7,
        )
        self._verify_forward_shape(inputs)

    def _verify_forward_shape(self, inputs):
        inputs_xpu = _move_to_device(inputs, "xpu")
        weights = inputs_xpu["weights"]
        weights[0].requires_grad_(True)
        if weights[1].numel() > 0:
            weights[1].requires_grad_(True)

        result = torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_pt2(
            placeholder_autograd_tensor=inputs_xpu["placeholder_autograd_tensor"],
            weights=weights,
            D_offsets=inputs_xpu["D_offsets"],
            total_D=inputs_xpu["total_D"],
            max_D=inputs_xpu["max_D"],
            hash_size_cumsum=inputs_xpu["hash_size_cumsum"],
            total_hash_size_bits=inputs_xpu["total_hash_size_bits"],
            indices=inputs_xpu["indices"],
            offsets=inputs_xpu["offsets"],
            pooling_mode=inputs_xpu["pooling_mode"],
            indice_weights=inputs_xpu["indice_weights"],
            feature_requires_grad=inputs_xpu["feature_requires_grad"],
            output_dtype=inputs_xpu["output_dtype"],
            max_B=inputs_xpu["max_B"],
            max_B_feature_rank=inputs_xpu["max_B_feature_rank"],
            vbe_output_size=inputs_xpu["vbe_output_size"],
            aux_tensor=inputs_xpu["aux_tensor"],
            aux_int=inputs_xpu["aux_int"],
            aux_float=inputs_xpu["aux_float"],
            aux_bool=inputs_xpu["aux_bool"],
            learning_rate_tensor=inputs_xpu["learning_rate_tensor"].cpu(),
            momentum1=inputs_xpu["momentum1"],
            optim_int=inputs_xpu["optim_int"],
            optim_float=inputs_xpu["optim_float"],
        )
        torch.xpu.synchronize()

        total_indices = inputs["indices"].numel()
        D = inputs["max_D"]
        self.assertEqual(result.shape, (total_indices, D))


class TestSplitLookupBackwardUniqCacheLocations(unittest.TestCase):
    """
    Tests for the backward kernel with use_uniq_cache_locations=True.

    This flag indicates that lxu_cache_locations contain unique slot values
    (each cache slot used by at most one row) and indices are pre-sorted so
    natural order matches sorted linear index order. This exercises the
    lxu_cache_locations.size(0) > 0 code path with the uniq flag set.
    """

    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise unittest.SkipTest("XPU is not available")

    def setUp(self):
        torch.xpu.synchronize()

    def tearDown(self):
        torch.xpu.empty_cache()
        torch.xpu.synchronize()

    def _run_forward_backward(self, inputs_xpu, grad_output_xpu):
        weights = inputs_xpu["weights"]
        weights[0].requires_grad_(True)
        if weights[1].numel() > 0:
            weights[1].requires_grad_(True)

        result = torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_pt2(
            placeholder_autograd_tensor=inputs_xpu["placeholder_autograd_tensor"],
            weights=weights,
            D_offsets=inputs_xpu["D_offsets"],
            total_D=inputs_xpu["total_D"],
            max_D=inputs_xpu["max_D"],
            hash_size_cumsum=inputs_xpu["hash_size_cumsum"],
            total_hash_size_bits=inputs_xpu["total_hash_size_bits"],
            indices=inputs_xpu["indices"],
            offsets=inputs_xpu["offsets"],
            pooling_mode=inputs_xpu["pooling_mode"],
            indice_weights=inputs_xpu["indice_weights"],
            feature_requires_grad=inputs_xpu["feature_requires_grad"],
            output_dtype=inputs_xpu["output_dtype"],
            max_B=inputs_xpu["max_B"],
            max_B_feature_rank=inputs_xpu["max_B_feature_rank"],
            vbe_output_size=inputs_xpu["vbe_output_size"],
            aux_tensor=inputs_xpu["aux_tensor"],
            aux_int=inputs_xpu["aux_int"],
            aux_float=inputs_xpu["aux_float"],
            aux_bool=inputs_xpu["aux_bool"],
            learning_rate_tensor=inputs_xpu["learning_rate_tensor"].cpu(),
            momentum1=inputs_xpu["momentum1"],
            optim_int=inputs_xpu["optim_int"],
            optim_float=inputs_xpu["optim_float"],
        )

        result.backward(grad_output_xpu)
        torch.xpu.synchronize()
        return result

    def _verify_backward(self, inputs, atol, rtol):
        total_indices = inputs["indices"].numel()
        D = inputs["max_D"]
        output_dtype_enum = SparseType(inputs["output_dtype"])
        if output_dtype_enum == SparseType.FP32:
            grad_dtype = torch.float32
        elif output_dtype_enum == SparseType.FP16:
            grad_dtype = torch.float16
        else:
            grad_dtype = torch.bfloat16

        torch.manual_seed(1234)
        grad_output = torch.randn(total_indices, D, dtype=grad_dtype)

        flat_weights = _get_flattened_weights(inputs)
        ref_weights, ref_momentum = _reference_adagrad_update(
            flat_weights=flat_weights,
            momentum1=inputs["momentum1"][0].clone(),
            grad_output=grad_output,
            indices=inputs["indices"],
            offsets=inputs["offsets"],
            hash_size_cumsum=inputs["hash_size_cumsum"],
            D_offsets=inputs["D_offsets"],
            learning_rate=inputs["learning_rate_tensor"].item(),
            eps=inputs["optim_float"][0],
        )
        ref_dev, ref_uvm = _scatter_weights_back(inputs, ref_weights)

        inputs_xpu = _move_to_device(copy.deepcopy(inputs), "xpu")
        grad_xpu = grad_output.to("xpu")
        self._run_forward_backward(inputs_xpu, grad_xpu)

        actual_dev = inputs_xpu["weights"][0].cpu().float()
        if actual_dev.numel() > 0:
            torch.testing.assert_close(
                actual_dev, ref_dev, atol=atol, rtol=rtol,
                msg="dev_weights update mismatch vs reference AdaGrad",
            )

        placements = inputs["weights"][2].tolist()
        has_caching = any(p == PlacementType.MANAGED_CACHING for p in placements)

        if has_caching:
            self._verify_caching_weights(
                inputs, inputs_xpu, ref_uvm, atol, rtol
            )
        else:
            actual_uvm = inputs_xpu["weights"][1].cpu().float()
            if actual_uvm.numel() > 0:
                torch.testing.assert_close(
                    actual_uvm, ref_uvm, atol=atol, rtol=rtol,
                    msg="uvm_weights update mismatch vs reference AdaGrad",
                )

        actual_momentum = inputs_xpu["momentum1"][0].cpu().float()
        torch.testing.assert_close(
            actual_momentum, ref_momentum, atol=atol, rtol=rtol,
            msg="Momentum update mismatch vs reference AdaGrad",
        )

    def _verify_caching_weights(self, inputs, inputs_xpu, ref_uvm, atol, rtol):
        placements = inputs["weights"][2].tolist()
        offsets_list = inputs["weights"][3].tolist()
        hash_size_cumsum = inputs["hash_size_cumsum"]
        D_offsets = inputs["D_offsets"]
        T = len(placements)

        actual_uvm = inputs_xpu["weights"][1].cpu().float()
        actual_cache = inputs_xpu["weights"][4].cpu().float()
        lxu_cache_locations = inputs["aux_tensor"][3]
        indices = inputs["indices"]
        input_offsets = inputs["offsets"]
        B = (input_offsets.numel() - 1) // T

        for t in range(T):
            if placements[t] != PlacementType.MANAGED_CACHING:
                continue

            D = (D_offsets[t + 1] - D_offsets[t]).item()
            num_rows = (hash_size_cumsum[t + 1] - hash_size_cumsum[t]).item()
            buf_offset = offsets_list[t]

            for row in range(num_rows):
                ref_start = buf_offset + row * D
                ref_row = ref_uvm[ref_start:ref_start + D]

                cache_slot = -1
                table_start_idx = t * B
                for b in range(B):
                    bag_idx = table_start_idx + b
                    idx_start = input_offsets[bag_idx].item()
                    idx_end = input_offsets[bag_idx + 1].item()
                    for l_idx in range(idx_start, idx_end):
                        if indices[l_idx].item() == row:
                            loc = lxu_cache_locations[l_idx].item()
                            if loc >= 0:
                                cache_slot = loc

                if cache_slot >= 0:
                    actual_row = actual_cache[cache_slot, :D]
                else:
                    actual_row = actual_uvm[ref_start:ref_start + D]

                torch.testing.assert_close(
                    actual_row, ref_row, atol=atol, rtol=rtol,
                    msg=f"MANAGED_CACHING table {t} row {row} mismatch",
                )

    # ===================================================================
    # use_uniq_cache_locations=True, mixed placements (non-homogeneous)
    # Exercises lxu_cache_locations.size(0) > 0 with unique cache slots
    # ===================================================================

    def test_uniq_cache_mixed_device_caching_fp32(self):
        """Mixed DEVICE + MANAGED_CACHING, unique cache locations, FP32."""
        inputs = _generate_uniq_cache_locations_data(
            T=4, B=16, D=64, num_rows_per_table=10, L=6,
            placements=[0, 2, 0, 2],
            output_dtype=SparseType.FP32,
            num_cache_slots=64,
            use_homogeneous_placements=False,
            seed=1000,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_uniq_cache_mixed_device_caching_fp16(self):
        """Mixed DEVICE + MANAGED_CACHING, unique cache locations, FP16."""
        inputs = _generate_uniq_cache_locations_data(
            T=4, B=16, D=64, num_rows_per_table=10, L=6,
            placements=[2, 0, 2, 1],
            output_dtype=SparseType.FP16,
            num_cache_slots=64,
            use_homogeneous_placements=False,
            seed=1003,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    def test_uniq_cache_mixed_device_caching_bf16(self):
        """Mixed DEVICE + MANAGED_CACHING, unique cache locations, BF16."""
        inputs = _generate_uniq_cache_locations_data(
            T=4, B=16, D=128, num_rows_per_table=10, L=6,
            placements=[0, 2, 1, 2],
            output_dtype=SparseType.BF16,
            num_cache_slots=64,
            use_homogeneous_placements=False,
            seed=1005,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    # ===================================================================
    # use_uniq_cache_locations=True, all three placement types
    # Maximum divergence between lxu_cache_locations sort order and
    # sorted_linear_indices sort order
    # ===================================================================

    def test_uniq_cache_three_placements_fp32(self):
        """DEVICE + MANAGED + MANAGED_CACHING, unique cache locations."""
        inputs = _generate_uniq_cache_locations_data(
            T=6, B=16, D=64, num_rows_per_table=10, L=5,
            placements=[0, 2, 1, 2, 0, 2],
            output_dtype=SparseType.FP32,
            num_cache_slots=128,
            use_homogeneous_placements=False,
            seed=1002,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_uniq_cache_three_placements_fp16(self):
        """DEVICE + MANAGED + MANAGED_CACHING, unique cache, FP16."""
        inputs = _generate_uniq_cache_locations_data(
            T=6, B=16, D=64, num_rows_per_table=10, L=5,
            placements=[2, 0, 1, 0, 2, 1],
            output_dtype=SparseType.FP16,
            num_cache_slots=128,
            use_homogeneous_placements=False,
            seed=1006,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    # ===================================================================
    # use_uniq_cache_locations=True, all MANAGED_CACHING (non-homogeneous)
    # ===================================================================

    def test_uniq_cache_all_caching_non_homogeneous_fp32(self):
        """All MANAGED_CACHING with use_homogeneous=False, unique cache."""
        inputs = _generate_uniq_cache_locations_data(
            T=3, B=32, D=128, num_rows_per_table=10, L=8,
            placements=[2, 2, 2],
            output_dtype=SparseType.FP32,
            num_cache_slots=128,
            use_homogeneous_placements=False,
            seed=1001,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_uniq_cache_all_caching_homogeneous_fp32(self):
        """All MANAGED_CACHING with use_homogeneous=True, unique cache."""
        inputs = _generate_uniq_cache_locations_data(
            T=4, B=16, D=64, num_rows_per_table=10, L=6,
            placements=[2, 2, 2, 2],
            output_dtype=SparseType.FP32,
            num_cache_slots=64,
            use_homogeneous_placements=True,
            seed=1007,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    # ===================================================================
    # use_uniq_cache_locations=True, large embedding dimension
    # ===================================================================

    def test_uniq_cache_large_D(self):
        """Large embedding dimension with unique cache locations."""
        inputs = _generate_uniq_cache_locations_data(
            T=3, B=8, D=256, num_rows_per_table=10, L=5,
            placements=[2, 0, 2],
            output_dtype=SparseType.FP32,
            num_cache_slots=64,
            use_homogeneous_placements=False,
            seed=1004,
        )
        self._verify_backward(inputs, atol=1e-3, rtol=1e-3)

    def test_uniq_cache_large_D_fp16(self):
        """Large D with unique cache locations, FP16."""
        inputs = _generate_uniq_cache_locations_data(
            T=3, B=8, D=512, num_rows_per_table=10, L=5,
            placements=[0, 2, 2],
            output_dtype=SparseType.FP16,
            num_cache_slots=64,
            use_homogeneous_placements=False,
            seed=1008,
        )
        self._verify_backward(inputs, atol=1e-1, rtol=1e-1)

    # ===================================================================
    # use_uniq_cache_locations=True, determinism
    # ===================================================================

    def test_uniq_cache_determinism(self):
        """Determinism with unique cache locations, mixed placements."""
        inputs = _generate_uniq_cache_locations_data(
            T=4, B=16, D=64, num_rows_per_table=10, L=6,
            placements=[0, 2, 0, 2],
            output_dtype=SparseType.FP32,
            num_cache_slots=64,
            use_homogeneous_placements=False,
            seed=1009,
        )
        total_indices = inputs["indices"].numel()
        D = inputs["max_D"]
        torch.manual_seed(700)
        grad_output = torch.randn(total_indices, D, dtype=torch.float32)

        results_dev = []
        results_uvm = []
        results_cache = []
        for _ in range(3):
            inputs_copy = copy.deepcopy(inputs)
            inputs_xpu = _move_to_device(inputs_copy, "xpu")
            grad_xpu = grad_output.to("xpu")
            self._run_forward_backward(inputs_xpu, grad_xpu)
            results_dev.append(inputs_xpu["weights"][0].cpu().clone())
            results_uvm.append(inputs_xpu["weights"][1].cpu().clone())
            results_cache.append(inputs_xpu["weights"][4].cpu().clone())

        for i in range(1, len(results_dev)):
            if results_dev[0].numel() > 0:
                torch.testing.assert_close(
                    results_dev[0], results_dev[i], atol=0, rtol=0,
                    msg=f"dev_weights: run {i} differs from run 0",
                )
            if results_uvm[0].numel() > 0:
                torch.testing.assert_close(
                    results_uvm[0], results_uvm[i], atol=0, rtol=0,
                    msg=f"uvm_weights: run {i} differs from run 0",
                )
            if results_cache[0].numel() > 0:
                torch.testing.assert_close(
                    results_cache[0], results_cache[i], atol=0, rtol=0,
                    msg=f"lxu_cache_weights: run {i} differs from run 0",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
