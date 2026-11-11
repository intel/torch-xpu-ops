# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestEmbeddingBackwardLargeSegments(TestCase):
    def test_embedding_dense_backward_large_num_segments(self):
        # Regression test for
        # "The number of work-items in each dimension of a work-group cannot
        # exceed {1024, 1024, 1024}".
        #
        # embedding_dense_backward's deterministic path launches
        # krn_partials_per_segment/krn_partial_segment_offset with a
        # single work-group sized by `num_of_segments` (the number of
        # distinct indices). When the number of distinct indices exceeds the
        # device's max work-group size (commonly 1024), the launch used to
        # fail with a RuntimeError instead of splitting the work across
        # multiple work-groups.
        num_weights = 4096
        embedding_dim = 8
        # More unique indices than any current device's max work-group size,
        # so num_of_segments > 1024.
        num_indices = 2000

        indices = torch.arange(num_indices, dtype=torch.long) % num_weights
        # Make sure we exceed 1024 unique indices.
        self.assertGreater(indices.unique().numel(), 1024)

        weight_cpu = torch.randn(
            num_weights, embedding_dim, dtype=torch.float32, requires_grad=True
        )
        grad_output_cpu = torch.randn(num_indices, embedding_dim)

        out_cpu = torch.nn.functional.embedding(indices, weight_cpu)
        out_cpu.backward(grad_output_cpu)
        expected_grad = weight_cpu.grad.clone()

        weight_xpu = weight_cpu.detach().to("xpu").requires_grad_()
        indices_xpu = indices.to("xpu")
        grad_output_xpu = grad_output_cpu.to("xpu")

        out_xpu = torch.nn.functional.embedding(indices_xpu, weight_xpu)
        out_xpu.backward(grad_output_xpu)

        self.assertEqual(weight_xpu.grad.cpu(), expected_grad)

    def test_embedding_dense_backward_large_num_segments_scale_grad_by_freq(self):
        # Same as above but with scale_grad_by_freq=True, which additionally
        # exercises the `count` tensor path in the deterministic backward
        # kernel.
        num_weights = 4096
        embedding_dim = 8
        num_indices = 2000

        indices = torch.arange(num_indices, dtype=torch.long) % num_weights
        self.assertGreater(indices.unique().numel(), 1024)

        weight_cpu = torch.randn(
            num_weights, embedding_dim, dtype=torch.float32, requires_grad=True
        )
        grad_output_cpu = torch.randn(num_indices, embedding_dim)

        out_cpu = torch.nn.functional.embedding(
            indices, weight_cpu, scale_grad_by_freq=True
        )
        out_cpu.backward(grad_output_cpu)
        expected_grad = weight_cpu.grad.clone()

        weight_xpu = weight_cpu.detach().to("xpu").requires_grad_()
        indices_xpu = indices.to("xpu")
        grad_output_xpu = grad_output_cpu.to("xpu")

        out_xpu = torch.nn.functional.embedding(
            indices_xpu, weight_xpu, scale_grad_by_freq=True
        )
        out_xpu.backward(grad_output_xpu)

        self.assertEqual(weight_xpu.grad.cpu(), expected_grad)


if __name__ == "__main__":
    run_tests()
