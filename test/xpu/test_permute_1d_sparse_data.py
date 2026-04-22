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

"""
Test suite for permute_1D_sparse_data custom operator

This module tests the correctness of the permute_1D_sparse_data operator
on both CPU and XPU (Intel GPU) devices.

Tests are designed to match TorchRec/DLRM sparse embedding scenarios.
"""

import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests
from typing import Optional, Tuple

# define fbgemm ops schemas here since we cannot register them in torch-xpu-ops.
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


_safe_define("permute_1D_sparse_data(Tensor permute, Tensor lengths, Tensor indices, Tensor? weights=None, int? permuted_lengths_sum=None) -> (Tensor, Tensor, Tensor?)")

# Reproducible seed for all random tests
SEED = 42


def compute_reference_permute_1d_sparse_data(
    permute: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Reference implementation of permute_1D_sparse_data using pure Python/PyTorch
    
    This is the ground truth implementation used for testing.
    
    Args:
        permute: Permutation indices tensor
        lengths: Segment lengths tensor
        indices: Concatenated values tensor
        weights: Optional weights tensor
        
    Returns:
        Tuple of (permuted_lengths, permuted_indices, permuted_weights)
    """
    # Move to CPU for reference computation
    permute_cpu = permute.cpu()
    lengths_cpu = lengths.cpu()
    indices_cpu = indices.cpu()
    weights_cpu = weights.cpu() if weights is not None else None
    
    permuted_lengths_size = permute_cpu.numel()
    
    # Compute permuted lengths
    permuted_lengths = torch.empty(permuted_lengths_size, dtype=lengths_cpu.dtype)
    for i in range(permuted_lengths_size):
        permuted_lengths[i] = lengths_cpu[permute_cpu[i]]
    
    # Compute input offsets (exclusive cumsum)
    input_offsets = torch.zeros(lengths_cpu.numel() + 1, dtype=torch.long)
    for i in range(lengths_cpu.numel()):
        input_offsets[i + 1] = input_offsets[i] + lengths_cpu[i]
    
    # Compute output offsets (exclusive cumsum)
    output_offsets = torch.zeros(permuted_lengths_size + 1, dtype=torch.long)
    for i in range(permuted_lengths_size):
        output_offsets[i + 1] = output_offsets[i] + permuted_lengths[i]
    
    # Compute permuted indices size
    permuted_indices_size = output_offsets[permuted_lengths_size].item()
    
    # Allocate output tensors
    permuted_indices = torch.empty(permuted_indices_size, dtype=indices_cpu.dtype)
    permuted_weights = torch.empty(permuted_indices_size, dtype=weights_cpu.dtype) if weights_cpu is not None else None
    
    # Copy data
    for j in range(permuted_lengths_size):
        segment_length = permuted_lengths[j].item()
        input_start = input_offsets[permute_cpu[j]].item()
        output_start = output_offsets[j].item()
        
        for i in range(segment_length):
            permuted_indices[output_start + i] = indices_cpu[input_start + i]
            if permuted_weights is not None:
                permuted_weights[output_start + i] = weights_cpu[input_start + i]
    
    # Move back to original device
    device = permute.device
    permuted_lengths = permuted_lengths.to(device)
    permuted_indices = permuted_indices.to(device)
    if permuted_weights is not None:
        permuted_weights = permuted_weights.to(device)
    
    return permuted_lengths, permuted_indices, permuted_weights


def generate_sparse_data(
    num_segments: int,
    avg_segment_length: int,
    device: str = "cpu",
    lengths_dtype: torch.dtype = torch.int32,
    indices_dtype: torch.dtype = torch.int64,
    weights_dtype: Optional[torch.dtype] = None,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate random sparse data for testing
    
    Args:
        num_segments: Number of segments
        avg_segment_length: Average length of each segment
        device: Device to place tensors on
        lengths_dtype: Data type for lengths tensor
        indices_dtype: Data type for indices tensor
        weights_dtype: Data type for weights tensor (None to skip weights)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (permute, lengths, indices, weights)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate random lengths (Poisson-like distribution around avg_segment_length)
    lengths = torch.randint(
        max(1, avg_segment_length - 5),
        avg_segment_length + 6,
        (num_segments,),
        dtype=lengths_dtype,
        device=device
    )
    
    # Total number of values
    total_values = lengths.sum().item()
    
    # Generate random indices
    indices = torch.randint(0, 10000, (total_values,), dtype=indices_dtype, device=device)
    
    # Generate random permutation
    permute = torch.randperm(num_segments, dtype=torch.int32, device=device)
    
    # Generate weights if requested
    weights = None
    if weights_dtype is not None:
        weights = torch.randn(total_values, dtype=weights_dtype, device=device)
    
    return permute, lengths, indices, weights


class TestPermute1DSparseData(TestCase):
    """Test cases for permute_1D_sparse_data operator"""
    
    def _test_correctness(self, device: str, lengths_dtype: torch.dtype, 
                          indices_dtype: torch.dtype, with_weights: bool = False):
        """
        Helper method to test correctness on a specific device and dtype combination
        """
        weights_dtype = torch.float32 if with_weights else None
        
        # Test case 1: Basic permutation [2, 0, 1]
        permute = torch.tensor([2, 0, 1], dtype=torch.int32, device=device)
        lengths = torch.tensor([3, 2, 4], dtype=lengths_dtype, device=device)
        indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=indices_dtype, device=device)
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
                               dtype=torch.float32, device=device) if with_weights else None
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute, lengths, indices, weights)
        expected = compute_reference_permute_1d_sparse_data(permute, lengths, indices, weights)
        
        torch.testing.assert_close(result[0], expected[0])
        torch.testing.assert_close(result[1], expected[1])
        if with_weights:
            torch.testing.assert_close(result[2], expected[2])
        
        # Verify expected values manually
        expected_lengths = torch.tensor([4, 3, 2], dtype=lengths_dtype, device=device)
        expected_indices = torch.tensor([5, 6, 7, 8, 0, 1, 2, 3, 4], dtype=indices_dtype, device=device)
        torch.testing.assert_close(result[0], expected_lengths)
        torch.testing.assert_close(result[1], expected_indices)
        
        # Test case 2: Identity permutation
        n_segments = 10
        permute_id = torch.arange(n_segments, dtype=torch.int32, device=device)
        lengths_id = torch.randint(1, 5, (n_segments,), dtype=lengths_dtype, device=device)
        total = lengths_id.sum().item()
        indices_id = torch.arange(total, dtype=indices_dtype, device=device)
        weights_id = torch.randn(total, dtype=torch.float32, device=device) if with_weights else None
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute_id, lengths_id, indices_id, weights_id)
        
        torch.testing.assert_close(result[0], lengths_id)
        torch.testing.assert_close(result[1], indices_id)
        if with_weights:
            torch.testing.assert_close(result[2], weights_id)
        
        # Test case 3: Reverse permutation
        n_segments = 5
        permute_rev = torch.arange(n_segments - 1, -1, -1, dtype=torch.int32, device=device)
        lengths_rev = torch.tensor([2, 3, 1, 4, 2], dtype=lengths_dtype, device=device)
        total = lengths_rev.sum().item()
        indices_rev = torch.arange(total, dtype=indices_dtype, device=device)
        weights_rev = torch.randn(total, dtype=torch.float32, device=device) if with_weights else None
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute_rev, lengths_rev, indices_rev, weights_rev)
        expected = compute_reference_permute_1d_sparse_data(permute_rev, lengths_rev, indices_rev, weights_rev)
        
        torch.testing.assert_close(result[0], expected[0])
        torch.testing.assert_close(result[1], expected[1])
        if with_weights:
            torch.testing.assert_close(result[2], expected[2])
        
        # Test case 4: Random medium-sized data
        permute, lengths, indices, weights = generate_sparse_data(
            num_segments=100,
            avg_segment_length=20,
            device=device,
            lengths_dtype=lengths_dtype,
            indices_dtype=indices_dtype,
            weights_dtype=weights_dtype,
            seed=SEED
        )
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute, lengths, indices, weights)
        expected = compute_reference_permute_1d_sparse_data(permute, lengths, indices, weights)
        
        torch.testing.assert_close(result[0], expected[0])
        torch.testing.assert_close(result[1], expected[1])
        if with_weights:
            torch.testing.assert_close(result[2], expected[2])
        
        # Test case 5: Large-scale data (DLRM-like)
        permute, lengths, indices, weights = generate_sparse_data(
            num_segments=1000,
            avg_segment_length=40,
            device=device,
            lengths_dtype=lengths_dtype,
            indices_dtype=indices_dtype,
            weights_dtype=weights_dtype,
            seed=SEED + 1
        )
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute, lengths, indices, weights)
        expected = compute_reference_permute_1d_sparse_data(permute, lengths, indices, weights)
        
        torch.testing.assert_close(result[0], expected[0])
        torch.testing.assert_close(result[1], expected[1])
        if with_weights:
            torch.testing.assert_close(result[2], expected[2])
    
    def _test_edge_cases(self, device: str):
        """Test edge cases"""
        # Test case 1: Empty permute
        permute_empty = torch.tensor([], dtype=torch.int32, device=device)
        lengths = torch.tensor([3, 2], dtype=torch.int32, device=device)
        indices = torch.arange(5, dtype=torch.int64, device=device)
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute_empty, lengths, indices)
        self.assertEqual(result[0].numel(), 0)
        
        # Test case 2: Single segment
        permute_single = torch.tensor([0], dtype=torch.int32, device=device)
        lengths_single = torch.tensor([5], dtype=torch.int32, device=device)
        indices_single = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64, device=device)
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute_single, lengths_single, indices_single)
        torch.testing.assert_close(result[0], lengths_single)
        torch.testing.assert_close(result[1], indices_single)
        
        # Test case 3: Zero-length segments
        permute_zero = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
        lengths_zero = torch.tensor([0, 3, 0], dtype=torch.int32, device=device)
        indices_zero = torch.tensor([1, 2, 3], dtype=torch.int64, device=device)
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute_zero, lengths_zero, indices_zero)
        expected = compute_reference_permute_1d_sparse_data(permute_zero, lengths_zero, indices_zero)
        
        torch.testing.assert_close(result[0], expected[0])
        torch.testing.assert_close(result[1], expected[1])
        
        # Test case 4: Permutation with duplicates (replication)
        permute_dup = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
        lengths_dup = torch.tensor([2, 3], dtype=torch.int32, device=device)
        indices_dup = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64, device=device)
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute_dup, lengths_dup, indices_dup)
        expected_lengths = torch.tensor([2, 2, 3], dtype=torch.int32, device=device)
        expected_indices = torch.tensor([10, 20, 10, 20, 30, 40, 50], dtype=torch.int64, device=device)
        
        torch.testing.assert_close(result[0], expected_lengths)
        torch.testing.assert_close(result[1], expected_indices)
    
    def _test_with_permuted_lengths_sum(self, device: str):
        """Test with precomputed permuted_lengths_sum to avoid sync"""
        permute = torch.tensor([2, 0, 1], dtype=torch.int32, device=device)
        lengths = torch.tensor([3, 2, 4], dtype=torch.int32, device=device)
        indices = torch.arange(9, dtype=torch.int64, device=device)
        
        # Precompute the sum: permuted_lengths = [4, 3, 2], sum = 9
        permuted_lengths_sum = 9
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(
            permute, lengths, indices, None, permuted_lengths_sum
        )
        expected = compute_reference_permute_1d_sparse_data(permute, lengths, indices)
        
        torch.testing.assert_close(result[0], expected[0])
        torch.testing.assert_close(result[1], expected[1])
    
    # ========================
    # XPU Tests
    # ========================
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_correctness_xpu_int32_int64(self):
        """Test XPU with int32 lengths and int64 indices"""
        self._test_correctness("xpu", torch.int32, torch.int64)
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_correctness_xpu_int64_int64(self):
        """Test XPU with int64 lengths and int64 indices"""
        self._test_correctness("xpu", torch.int64, torch.int64)
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_correctness_xpu_int32_int32(self):
        """Test XPU with int32 lengths and int32 indices"""
        self._test_correctness("xpu", torch.int32, torch.int32)
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_correctness_xpu_with_weights(self):
        """Test XPU with weights"""
        self._test_correctness("xpu", torch.int32, torch.int64, with_weights=True)
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_edge_cases_xpu(self):
        """Test edge cases on XPU"""
        self._test_edge_cases("xpu")
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_with_permuted_lengths_sum_xpu(self):
        """Test with precomputed permuted_lengths_sum on XPU"""
        self._test_with_permuted_lengths_sum("xpu")

class TestPermute1DSparseDataDLRMScale(TestCase):
    """
    Performance-oriented tests at DLRM production scale
    """
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_dlrm_v2_scale(self):
        """Test at DLRMv2 typical scale"""
        # DLRMv2 configuration
        num_features = 26
        batch_size = 512
        avg_pooling_factor = 40
        
        total_segments = num_features * batch_size  # 13,312
        
        permute, lengths, indices, weights = generate_sparse_data(
            num_segments=total_segments,
            avg_segment_length=avg_pooling_factor,
            device="xpu",
            lengths_dtype=torch.int32,
            indices_dtype=torch.int64,
            weights_dtype=torch.float32,
            seed=SEED
        )
        
        # Run and verify
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute, lengths, indices, weights)
        expected = compute_reference_permute_1d_sparse_data(permute, lengths, indices, weights)
        
        torch.testing.assert_close(result[0], expected[0])
        torch.testing.assert_close(result[1], expected[1])
        torch.testing.assert_close(result[2], expected[2])
    
    @unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
    def test_large_segments(self):
        """Test with large segment sizes"""
        permute, lengths, indices, _ = generate_sparse_data(
            num_segments=100,
            avg_segment_length=1000,  # Large segments
            device="xpu",
            lengths_dtype=torch.int32,
            indices_dtype=torch.int64,
            seed=SEED
        )
        
        result = torch.ops.fbgemm.permute_1D_sparse_data(permute, lengths, indices)
        expected = compute_reference_permute_1d_sparse_data(permute, lengths, indices)
        
        torch.testing.assert_close(result[0], expected[0])
        torch.testing.assert_close(result[1], expected[1])


if __name__ == "__main__":
    run_tests()
