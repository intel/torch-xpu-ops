"""
Reproducer for https://github.com/pytorch/pytorch/issues/181699

test_max_reads_limits_fusion (OverFusionTest) in
test/inductor/test_mix_order_reduction.py fails on XPU CI.

The test verifies that mix_order_reduction max_reads limits over-fusion in
a transformer backward pass (GQA + QK-norm + squared leaky-relu MLP).

On XPU, the test does not produce the expected metrics
(rejected_mix_order_reduction_fusion > 0), causing assertion failures.

Upstream fix: added @skipIfXpu to test_max_reads_limits_fusion in pytorch/pytorch.
This repro tracks the issue for future re-enablement on XPU.

Run:
    pytest test/repro/test_mix_order_reduction_over_fusion.py
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.mark.skip(
    reason="Skipped on XPU: https://github.com/pytorch/pytorch/issues/181699"
)
@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
def test_max_reads_limits_fusion_xpu():
    """
    Verify that max_reads limits over-fusion in a transformer backward pass
    on XPU without disabling mix_order_reduction entirely.

    Uses the exact model pattern from #179423: GQA attention with QK-norm
    and squared leaky-relu MLP. The QK-norm creates extra intermediate
    buffers in the backward pass that push read counts above the threshold.

    This test is currently expected to fail on XPU because
    rejected_mix_order_reduction_fusion is not triggered as expected.
    """
    import torch._inductor.config as inductor_config
    from torch._dynamo.utils import same
    from torch._inductor import metrics

    metrics.reset()

    num_heads = 8
    num_kv_heads = 4
    dim = 512
    head_dim = dim // num_heads

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_q = nn.Linear(dim, dim, bias=False)
            self.c_k = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
            self.c_v = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)

        def forward(self, x):
            B, T, D = x.shape
            q = self.c_q(x).reshape(B, T, num_heads, head_dim)
            k = self.c_k(x).reshape(B, T, num_kv_heads, head_dim)
            v = self.c_v(x).reshape(B, T, num_kv_heads, head_dim)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            q = q.transpose(1, 2)
            k = k.transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
            v = v.transpose(1, 2).repeat_interleave(num_heads // num_kv_heads, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.proj(y.transpose(1, 2).reshape(B, T, D))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_norm = nn.RMSNorm(dim)
            self.mlp_norm = nn.RMSNorm(dim)
            self.attn = Attention()
            self.fc1 = nn.Linear(dim, dim * 4, bias=False)
            self.fc2 = nn.Linear(dim * 4, dim, bias=False)

        def forward(self, x):
            x = x + self.attn(self.attn_norm(x))
            h = self.mlp_norm(x)
            x = x + self.fc2(
                F.leaky_relu(self.fc1(h), negative_slope=0.5).square()
            )
            return x

    model = nn.Sequential(*[Block() for _ in range(3)]).to("xpu").bfloat16()
    x = torch.randn(
        8, 2048, dim, device="xpu", dtype=torch.bfloat16, requires_grad=True
    )
    dy = torch.randn_like(x)

    out_ref = model(x)
    out_ref.backward(dy)
    grad_ref = x.grad.clone()
    x.grad = None

    with inductor_config.patch(
        {
            "triton.mix_order_reduction": True,
            "triton.mix_order_reduction_max_reads": 10,
        }
    ):
        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        out_act = compiled(x)
        out_act.backward(dy)
        grad_act = x.grad.clone()

    assert same(grad_ref, grad_act, tol=5e-2), (
        f"Gradient mismatch: ref mean={grad_ref.abs().mean()}, "
        f"act mean={grad_act.abs().mean()}"
    )
    assert metrics.codegen_mix_order_reduction > 0, (
        f"Expected codegen_mix_order_reduction > 0, "
        f"got {metrics.codegen_mix_order_reduction}"
    )
    assert metrics.rejected_mix_order_reduction_fusion > 0, (
        f"Expected rejected_mix_order_reduction_fusion > 0, "
        f"got {metrics.rejected_mix_order_reduction_fusion}"
    )
