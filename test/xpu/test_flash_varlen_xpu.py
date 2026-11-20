# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: sdpa"]
"""Varlen (cu_seqlens) forward of the XPU SYCL-TLA flash attention.

Calls aten::_flash_attention_forward with cum_seq_q/k, as
torch/nn/attention/varlen.py does, and checks out and lse against an fp32
per-document reference.
"""

import unittest

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase
from torch.testing._internal.common_xpu import PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU


def make_cu(lens, device):
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    return torch.tensor(cu, dtype=torch.int32, device=device)


def varlen_fwd(q, k, v, cu_q, cu_k, max_q, max_k, is_causal, **kwargs):
    # Same call as torch/nn/attention/varlen.py: causal means window (-1, 0).
    out, lse, _, _, _ = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        kwargs.pop("dropout_p", 0.0),
        is_causal,
        False,
        scale=kwargs.pop("scale", None),
        window_size_left=kwargs.pop("window_size_left", -1),
        window_size_right=kwargs.pop("window_size_right", 0 if is_causal else -1),
        **kwargs,
    )
    return out, lse


def reference(q, k, v, lens_q, lens_k, is_causal):
    """fp32 attention, one document at a time.

    Causal is bottom-right aligned (FA2): query i sees key j if
    j <= i + len_k - len_q. Rows that see no key get out = 0, lse = +inf.
    """
    device = q.device
    total_q, hq, d = q.shape
    hkv = k.shape[1]
    out = torch.zeros(total_q, hq, v.shape[2], dtype=torch.float32, device=device)
    lse = torch.full((hq, total_q), float("inf"), dtype=torch.float32, device=device)
    scale = d**-0.5
    sq = sk = 0
    for lq, lk in zip(lens_q, lens_k):
        if lq > 0 and lk > 0:
            qd = q[sq : sq + lq].float().transpose(0, 1)  # [Hq, lq, D]
            kd = k[sk : sk + lk].float().transpose(0, 1)  # [Hkv, lk, D]
            vd = v[sk : sk + lk].float().transpose(0, 1)
            kd = kd.repeat_interleave(hq // hkv, dim=0)  # GQA
            vd = vd.repeat_interleave(hq // hkv, dim=0)
            s = (qd @ kd.transpose(1, 2)) * scale  # [Hq, lq, lk]
            if is_causal:
                i = torch.arange(lq, device=device)[:, None]
                j = torch.arange(lk, device=device)[None, :]
                s = s.masked_fill(j > i + (lk - lq), float("-inf"))
            row_lse = torch.logsumexp(s, dim=-1)  # [Hq, lq], -inf if no key
            p = torch.exp(s - row_lse[..., None]).nan_to_num(0.0)
            out[sq : sq + lq] = (p @ vd).transpose(0, 1)
            lse[:, sq : sq + lq] = torch.where(
                torch.isinf(row_lse), float("inf"), row_lse
            )
        sq += lq
        sk += lk
    return out, lse


# Like the dense flash tests, this skips PVC (pytorch#178131). On PVC the kernels
# are correct when AOT-compiled with a recent IGC (2.30 passes, 2.11 does not).
@unittest.skipIf(
    not PLATFORM_SUPPORTS_FLASH_ATTENTION_XPU, "SYCL-TLA flash attention unsupported"
)
class TestFlashVarlen(TestCase):
    def _run(
        self,
        device,
        lens_q,
        lens_k=None,
        hq=2,
        hkv=None,
        d=128,
        dtype=torch.bfloat16,
        is_causal=True,
    ):
        lens_k = lens_q if lens_k is None else lens_k
        hkv = hq if hkv is None else hkv
        torch.manual_seed(0)
        q = torch.randn(sum(lens_q), hq, d, dtype=dtype, device=device)
        k = torch.randn(sum(lens_k), hkv, d, dtype=dtype, device=device)
        v = torch.randn(sum(lens_k), hkv, d, dtype=dtype, device=device)
        out, lse = varlen_fwd(
            q,
            k,
            v,
            make_cu(lens_q, device),
            make_cu(lens_k, device),
            max(lens_q),
            max(lens_k),
            is_causal,
        )
        ref_out, ref_lse = reference(q, k, v, lens_q, lens_k, is_causal)

        self.assertEqual(out.shape, q.shape)
        self.assertEqual(out.dtype, dtype)
        self.assertEqual(lse.shape, (hq, sum(lens_q)))
        self.assertEqual(lse.dtype, torch.float32)
        # P is rounded to 16 bits before the PV GEMM, so out is looser than lse.
        torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-3)

    @parametrize("is_causal", [False, True])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("d", [32, 64, 80, 96, 128, 192, 256])  # 80 is padded to 96
    def test_head_dims(self, device, d, dtype, is_causal):
        self._run(device, [5, 300, 129], d=d, dtype=dtype, is_causal=is_causal)

    @parametrize("is_causal", [False, True])
    def test_doc_lengths(self, device, is_causal):
        # Empty docs, length 1, tile boundaries (128) and non-multiples.
        self._run(device, [0, 1, 127, 128, 129, 0, 513, 7], is_causal=is_causal)

    def test_single_doc(self, device):
        self._run(device, [1000])

    @parametrize("is_causal", [False, True])
    def test_gqa(self, device, is_causal):
        self._run(device, [5, 300, 64], hq=4, hkv=2, is_causal=is_causal)
        self._run(device, [33, 200], hq=8, hkv=1, is_causal=is_causal)

    @parametrize("is_causal", [False, True])
    def test_q_k_lengths_differ(self, device, is_causal):
        # Causal is bottom-right per document. lq > lk leaves fully masked rows.
        self._run(device, [5, 300, 64], lens_k=[17, 300, 30], is_causal=is_causal)

    def test_decode_one_query(self, device):
        # max_seqlen_q == 1 turns causal off, as in FA2.
        self._run(device, [1, 1, 1], lens_k=[50, 200, 1])

    def test_matches_dense(self, device):
        # Equal-length docs: varlen must match the dense kernel.
        b, s, h, d = 3, 200, 2, 128
        torch.manual_seed(0)
        q, k, v = (
            torch.randn(b, s, h, d, dtype=torch.bfloat16, device=device)
            for _ in range(3)
        )
        dense_out, dense_lse, _, _, _ = torch.ops.aten._flash_attention_forward(
            q, k, v, None, None, s, s, 0.0, True, False
        )
        cu = make_cu([s] * b, device)
        out, lse = varlen_fwd(
            q.flatten(0, 1), k.flatten(0, 1), v.flatten(0, 1), cu, cu, s, s, True
        )
        torch.testing.assert_close(out, dense_out.flatten(0, 1), atol=0, rtol=0)
        # dense lse [B, H, S] -> varlen [H, B*S]
        torch.testing.assert_close(
            lse, dense_lse.transpose(0, 1).flatten(1, 2), atol=0, rtol=0
        )

    def test_custom_scale(self, device):
        q = torch.randn(305, 2, 128, dtype=torch.bfloat16, device=device)
        cu = make_cu([5, 300], device)
        out, lse = varlen_fwd(q, q, q, cu, cu, 300, 300, True, scale=0.05)
        ref_out, ref_lse = reference(
            q * (0.05 * 128**0.5), q, q, [5, 300], [5, 300], True
        )
        torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-3)

    def test_num_splits_one(self, device):
        q = torch.randn(305, 2, 128, dtype=torch.bfloat16, device=device)
        cu = make_cu([5, 300], device)
        varlen_fwd(q, q, q, cu, cu, 300, 300, True, num_splits=1)

    def test_empty_keys(self, device):
        # max_seqlen_k == 0: out = 0, lse = +inf (FA2).
        q = torch.randn(10, 2, 64, dtype=torch.bfloat16, device=device)
        k = torch.randn(0, 2, 64, dtype=torch.bfloat16, device=device)
        out, lse = varlen_fwd(
            q, k, k, make_cu([4, 6], device), make_cu([0, 0], device), 6, 0, False
        )
        self.assertEqual(out, torch.zeros_like(q))
        self.assertTrue(torch.isposinf(lse).all())

    def test_errors(self, device):
        q = torch.randn(305, 2, 128, dtype=torch.bfloat16, device=device)
        cu = make_cu([5, 300], device)
        with self.assertRaisesRegex(RuntimeError, "int32"):
            varlen_fwd(q, q, q, cu.long(), cu.long(), 300, 300, True)
        with self.assertRaisesRegex(RuntimeError, "dropout"):
            varlen_fwd(q, q, q, cu, cu, 300, 300, True, dropout_p=0.1)
        with self.assertRaisesRegex(RuntimeError, "num_splits"):
            varlen_fwd(q, q, q, cu, cu, 300, 300, True, num_splits=2)
        with self.assertRaises(RuntimeError):  # sliding window
            varlen_fwd(q, q, q, cu, cu, 300, 300, True, window_size_left=5)
        with self.assertRaisesRegex(RuntimeError, "seqused_k"):
            varlen_fwd(q, q, q, cu, cu, 300, 300, True, seqused_k=cu[1:].contiguous())


instantiate_device_type_tests(
    TestFlashVarlen, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
