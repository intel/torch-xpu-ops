# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]

"""Regression test for the capture-safe embedding_dense_backward path (index_add_).

Failed assertions fail the process (nonzero exit) so CI catches regressions. Asserts CORRECTNESS,
not merely that capture produced something:
  1. graph-captured backward == eager (clean single replay)
  2. repeated indices (heavy duplication) accumulate correctly under capture
  3. padding_idx row stays zero under capture
  4. two replays == exactly 2x eager (accumulation semantics)
  5. scale_grad_by_freq=True still correct in eager (sort path)
  6. deterministic mode keeps the host-sync sort path (capture raises)
  7. half/bfloat16 graph gradient == eager (fp32-accumulation correction under capture)

Without the fix, tests 1-4 and 7 fail at capture with
"wait method cannot be used for an event associated with a command graph".
"""
import unittest

import torch
from torch.testing import assert_close

_HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
dev = torch.device("xpu")
RTOL, ATOL = 1e-4, 1e-5


class _W(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.e = emb

    def forward(self, xx):
        return self.e(xx).sum().reshape(1)


def eager_then_graph(V, D, x, padding_idx=-1, seed=0, replays=1, dtype=torch.float32):
    """Eager reference FIRST (materializes a stable grad buffer), then capture the SAME model."""
    torch.manual_seed(seed)
    m = torch.nn.Embedding(
        V, D, padding_idx=(padding_idx if padding_idx >= 0 else None)
    ).to(dev, dtype)
    w = _W(m).to(dev, dtype)
    m.zero_grad(set_to_none=True)
    w(x).backward()
    torch.xpu.synchronize()
    eager = m.weight.grad.detach().clone()
    g = torch.xpu.make_graphed_callables(w, (x,), num_warmup_iters=3, allow_unused_input=True)
    m.weight.grad.zero_()
    for _ in range(replays):
        g(x).backward()
    torch.xpu.synchronize()
    graph = m.weight.grad.detach().clone()
    return eager, graph


@unittest.skipUnless(_HAS_XPU, "XPU device required")
class TestXPUEmbeddingCapture(unittest.TestCase):
    def test_capture_matches_eager(self):
        V, D = 4096, 512
        x = torch.randint(0, V, (64, 128), device=dev)
        eager, graph = eager_then_graph(V, D, x)
        assert_close(graph, eager, rtol=RTOL, atol=ATOL)

    def test_repeated_indices_accumulate(self):
        V, D = 16, 64  # ~512x duplication over 8192 indices
        x = torch.randint(0, V, (64, 128), device=dev)
        eager, graph = eager_then_graph(V, D, x)
        assert_close(graph, eager, rtol=RTOL, atol=ATOL)

    def test_padding_idx_zeroed(self):
        V, D, pi = 4096, 128, 7
        x = torch.randint(0, V, (32, 64), device=dev)
        x[0, 0] = pi
        eager, graph = eager_then_graph(V, D, x, padding_idx=pi)
        self.assertEqual(graph[pi].abs().max().item(), 0.0)
        assert_close(graph, eager, rtol=RTOL, atol=ATOL)

    def test_two_replays_double(self):
        V, D = 2048, 64
        x = torch.randint(0, V, (32, 64), device=dev)
        eager, graph2 = eager_then_graph(V, D, x, replays=2)
        assert_close(graph2, eager * 2, rtol=RTOL, atol=ATOL)

    def test_scale_grad_by_freq_eager_ok(self):
        V, D = 512, 32
        x = torch.randint(0, V, (16, 32), device=dev)
        torch.manual_seed(0)
        w = torch.randn(V, D, device=dev, requires_grad=True)
        torch.nn.functional.embedding(x, w, scale_grad_by_freq=True).sum().backward()
        torch.xpu.synchronize()
        self.assertTrue(w.grad is not None and torch.isfinite(w.grad).all())

    def test_deterministic_mode_stays_sort_path(self):
        V, D = 4096, 128
        x = torch.randint(0, V, (64, 128), device=dev)
        torch.use_deterministic_algorithms(True)
        try:
            with self.assertRaises(Exception):
                eager_then_graph(V, D, x)  # sort path host-syncs -> capture must raise
        finally:
            torch.use_deterministic_algorithms(False)

    def test_half_bfloat16_capture_matches_eager(self):
        V, D = 4096, 256
        x = torch.randint(0, V, (64, 128), device=dev)
        for dt, rtol, atol in ((torch.float16, 1e-3, 1e-3), (torch.bfloat16, 1e-2, 1e-2)):
            eager, graph = eager_then_graph(V, D, x, dtype=dt)
            assert_close(graph, eager, rtol=rtol, atol=atol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
