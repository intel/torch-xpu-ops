# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests, TestCase


class TestEmbeddingBagLargeIndex(TestCase):
    def _large_index_setup(self):
        large_index = 2**31
        weight = torch.empty((large_index + 1, 1), device="xpu", dtype=torch.float16)
        weight[large_index] = 3.0
        indices = torch.tensor([large_index], device="xpu", dtype=torch.int64)
        offsets = torch.tensor([0], device="xpu", dtype=torch.int64)
        per_sample_weights = torch.ones(
            1, device="xpu", dtype=torch.float16, requires_grad=True
        )
        return weight, indices, offsets, per_sample_weights

    def test_embedding_bag_per_sample_weights_grad_large_index(self):
        # To check if the gradient of per-sample-weights backward kernel
        # came back as 0 instead of the expected value when indices >= 2**31
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        prop = torch.xpu.get_device_properties(0)
        # weight is (2**31 + 1) x 1 fp16 = ~4.3 GiB.
        if prop.total_memory < 6 * 1024**3:
            self.skipTest("needs >= 6 GiB of device memory")

        weight, indices, offsets, per_sample_weights = self._large_index_setup()
        F.embedding_bag(
            indices,
            weight,
            offsets,
            mode="sum",
            per_sample_weights=per_sample_weights,
        ).sum().backward()

        self.assertEqual(
            per_sample_weights.grad,
            torch.tensor([3.0], device="xpu", dtype=torch.float16),
        )


if __name__ == "__main__":
    run_tests()
