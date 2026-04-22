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

import torch
import unittest
import random
from torch.testing._internal.common_utils import TestCase, run_tests

# define fbgemm ops schemas here since we cannot register them in torch-xpu-ops.
# otherwise, it will fail fbgemm lib due to duplicate schema registration.
# for user, they can import fbgemm_gpu first before accessing fbgemm ops on xpu.
lib = torch.library.Library("fbgemm", "DEF")  # noqa: TOR901

lib.define("jagged_index_select_2d_forward(Tensor values, Tensor indices, Tensor input_offsets, Tensor output_offsets, int num_dense_output_rows) -> Tensor")

class TestJaggedIndexSelect(TestCase):
    
    def setUp(self):
        super().setUp()
        random.seed(42)
        torch.manual_seed(42)

    def _jagged_index_select_ref(self, values, indices, input_offsets, output_offsets, num_dense_output_rows):
        """CPU Reference implementation"""
        values_cpu = values.cpu()
        indices_cpu = indices.cpu()
        input_offsets_cpu = input_offsets.cpu()
        output_offsets_cpu = output_offsets.cpu()
        
        num_cols = values.size(1)
        output = torch.zeros((num_dense_output_rows, num_cols), dtype=values.dtype)
        
        for i, idx in enumerate(indices_cpu):
            idx = idx.item()
            
            # Input range
            input_start = 0 if idx == 0 else input_offsets_cpu[idx - 1].item()
            input_end = input_offsets_cpu[idx].item()
            
            # Output range
            output_start = 0 if i == 0 else output_offsets_cpu[i - 1].item()
            output_end = output_offsets_cpu[i].item()
            
            # Copy
            seq_len = input_end - input_start
            out_len = output_end - output_start
            
            if seq_len != out_len:
                 raise RuntimeError(f"Length mismatch at index {i} (input idx {idx}): input len {seq_len}, output len {out_len}")
            
            if seq_len > 0:
                output[output_start:output_end] = values_cpu[input_start:input_end]
                
        return output.to(values.device)

    def _run_test(self, num_sequences, num_cols, num_selected, device="xpu", 
                  empty_input_segments=False, empty_output_segments=False, 
                  duplicates=False, dtype=torch.float32):
        
        if not torch.xpu.is_available() and device == "xpu":
            self.skipTest("XPU not available")

        # 1. Generate Input Jagged Tensor
        # Generate random lengths
        if empty_input_segments:
            # Mix of 0s and small numbers
            input_lengths = torch.randint(0, 5, (num_sequences,), device=device)
        else:
            input_lengths = torch.randint(1, 10, (num_sequences,), device=device)
            
        input_offsets = torch.cumsum(input_lengths, dim=0)
        total_input_rows = input_offsets[-1].item() if num_sequences > 0 else 0
        
        values = torch.randn((total_input_rows, num_cols), dtype=dtype, device=device)
        
        # 2. Generate Indices to Select
        if num_selected == 0:
            indices = torch.tensor([], dtype=torch.long, device=device)
        else:
            if duplicates:
                indices = torch.randint(0, num_sequences, (num_selected,), device=device)
            else:
                # Sample without replacement if possible, else with replacement (if num_selected > num_sequences)
                if num_selected <= num_sequences:
                    # randperm
                    indices = torch.randperm(num_sequences, device=device)[:num_selected]
                else:
                    indices = torch.randint(0, num_sequences, (num_selected,), device=device)

        # 3. Calculate Output Offsets based on selected indices
        # We need to match the lengths of the selected sequences
        selected_lengths = []
        if num_selected > 0:
            # We need to get lengths on CPU to construct offsets easily or use GPU torch.ops.fbgemm
            # Using CPU for test setup is fine
            input_lengths_cpu = input_lengths.cpu()
            indices_cpu = indices.cpu()
            for idx in indices_cpu:
                selected_lengths.append(input_lengths_cpu[idx].item())
        
        output_lengths = torch.tensor(selected_lengths, dtype=torch.long, device=device)
        output_offsets = torch.cumsum(output_lengths, dim=0)
        num_dense_output_rows = output_offsets[-1].item() if num_selected > 0 else 0
        
        # 4. Run Operator
        output = torch.ops.fbgemm.jagged_index_select_2d_forward(
            values, indices, input_offsets, output_offsets, num_dense_output_rows
        )
        
        # 5. Run Reference
        expected = self._jagged_index_select_ref(
            values, indices, input_offsets, output_offsets, num_dense_output_rows
        )
        
        # 6. Compare
        self.assertEqual(output.shape, expected.shape)
        torch.testing.assert_close(output, expected)

    def test_basic_correctness(self):
        self._run_test(num_sequences=10, num_cols=8, num_selected=5)

    def test_duplicates(self):
        self._run_test(num_sequences=10, num_cols=8, num_selected=20, duplicates=True)

    def test_empty_input_segments(self):
        self._run_test(num_sequences=20, num_cols=4, num_selected=10, empty_input_segments=True)
        
    def test_select_nothing(self):
        self._run_test(num_sequences=10, num_cols=4, num_selected=0)

    def test_select_all_reordered(self):
        # Select all but shuffled
        self._run_test(num_sequences=10, num_cols=4, num_selected=10, duplicates=False)

    def test_large_batch(self):
        self._run_test(num_sequences=1000, num_cols=32, num_selected=500)

    def test_very_large_batch(self):
        # Stress test size
        self._run_test(num_sequences=10000, num_cols=16, num_selected=10000)

    def test_single_sequence(self):
        self._run_test(num_sequences=1, num_cols=10, num_selected=1)

    def test_single_sequence_duplicate(self):
        self._run_test(num_sequences=1, num_cols=10, num_selected=5, duplicates=True)

    def test_varying_cols_small(self):
        self._run_test(num_sequences=10, num_cols=1, num_selected=5)

    def test_varying_cols_large(self):
        self._run_test(num_sequences=10, num_cols=128, num_selected=5)
        
    def test_varying_cols_odd(self):
        self._run_test(num_sequences=10, num_cols=7, num_selected=5)

    def test_input_offsets_zero_start(self):
        # Test implicitly handled by logic, but ensuring first element handling
        self._run_test(num_sequences=5, num_cols=4, num_selected=3)

    def test_values_dim_mismatch(self):
        # Renamed from test_output_offsets_mismatch_shape to be more accurate
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        
        values = torch.randn(10, 4, device="xpu")
        indices = torch.tensor([0], device="xpu")
        input_offsets = torch.tensor([10], device="xpu")
        output_offsets = torch.tensor([10], device="xpu")
        
        # Pass 1D values instead of 2D
        with self.assertRaisesRegex(RuntimeError, "values must be 2D tensor"):
             torch.ops.fbgemm.jagged_index_select_2d_forward(
                values.view(-1), indices, input_offsets, output_offsets, 10
            )

    def test_indices_dim_mismatch(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        values = torch.randn(10, 4, device="xpu")
        indices = torch.randn(2, 2, device="xpu") # 2D indices
        input_offsets = torch.tensor([10], device="xpu")
        output_offsets = torch.tensor([10], device="xpu")
        
        with self.assertRaisesRegex(RuntimeError, "indices must be 1D tensor"):
             torch.ops.fbgemm.jagged_index_select_2d_forward(
                values, indices, input_offsets, output_offsets, 10
            )

    def test_offsets_dim_mismatch(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        values = torch.randn(10, 4, device="xpu")
        indices = torch.tensor([0], device="xpu")
        # Pass 2D Long tensor
        input_offsets = torch.randint(0, 10, (2, 2), dtype=torch.long, device="xpu")
        output_offsets = torch.tensor([10], device="xpu")
        
        with self.assertRaisesRegex(RuntimeError, "input_offsets must be 1D tensor"):
             torch.ops.fbgemm.jagged_index_select_2d_forward(
                values, indices, input_offsets, output_offsets, 10
            )

    # Generate more tests to reach 20
    def test_case_16(self): self._run_test(50, 16, 25)
    def test_case_17(self): self._run_test(50, 16, 25, duplicates=True)
    def test_case_18(self): self._run_test(50, 16, 25, empty_input_segments=True)
    def test_case_19(self): self._run_test(10, 64, 10)
    def test_case_20(self): self._run_test(10, 64, 10, duplicates=True)
    def test_case_21(self): self._run_test(5, 256, 5)

if __name__ == "__main__":
    run_tests()
