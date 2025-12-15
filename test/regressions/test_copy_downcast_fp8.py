# Copyright 2020-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import TestCase


class TestTorchMethod(TestCase):
    def test_fp8_downcast_copy_float8_e4m3(self, dtype=torch.float8_e4m3fn):
        seed = 123
        torch.manual_seed(seed)
        tensor_fp32 = (
            torch.rand((32, 32), dtype=torch.float32, device=torch.device("xpu")) / 10
        )
        # print("tensor_bf16 = ", tensor_bf16)
        tensor_fp8 = tensor_fp32.to(dtype)
        # print("tensor_fp8 = ", tensor_fp8)
        # print("tensor_fp8_fp32 = ", tensor_fp8.float())
        self.assertEqual(tensor_fp32, tensor_fp8.float(), atol=1e-2, rtol=1e-2)

    def test_fp8_downcast_copy_float8_e5m2(self, dtype=torch.float8_e5m2):
        seed = 123
        torch.manual_seed(seed)
        tensor_fp16 = (
            torch.rand((64, 32), dtype=torch.float16, device=torch.device("xpu")) / 10
        )
        # print("tensor_bf16 = ", tensor_fp16)
        tensor_fp8 = tensor_fp16.to(dtype)
        # print("tensor_fp8 = ", tensor_fp8)
        # print("tensor_fp8_fp16 = ", tensor_fp8.half())
        self.assertEqual(tensor_fp16, tensor_fp8.half(), atol=1e-2, rtol=1e-2)

    def test_fp8_downcast_copy_float8_e4m3fnuz(self, dtype=torch.float8_e4m3fnuz):
        seed = 123
        torch.manual_seed(seed)
        tensor_bf16 = (
            torch.rand((32, 32), dtype=torch.bfloat16, device=torch.device("xpu")) / 10
        )
        # print("tensor_bf16 = ", tensor_bf16)
        tensor_fp8 = tensor_bf16.to(dtype)
        # print("tensor_fp8 = ", tensor_fp8)
        # print("tensor_fp8_bf16 = ", tensor_fp8.bfloat16())
        self.assertEqual(tensor_bf16, tensor_fp8.bfloat16(), atol=1e-2, rtol=1e-2)

    def test_fp8_downcast_copy_float8_e5m2fnuz(self, dtype=torch.float8_e5m2fnuz):
        seed = 123
        torch.manual_seed(seed)
        tensor_fp16 = (
            torch.rand((32, 32), dtype=torch.float16, device=torch.device("xpu")) / 10
        )
        # print("tensor_bf16 = ", tensor_fp16)
        tensor_fp8 = tensor_fp16.to(dtype)
        # print("tensor_fp8 = ", tensor_fp8)
        # print("tensor_fp8_fp16 = ", tensor_fp8.half())
        self.assertEqual(tensor_fp16, tensor_fp8.half(), atol=1e-2, rtol=1e-2)
