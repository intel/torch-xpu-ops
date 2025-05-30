# Owner(s): ["module: intel"]

import contextlib

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    from .xpu_test_utils import XPUPatchForImport


def get_device_capability(device=None):
    return (9, 0)


torch.cuda.get_device_capability = get_device_capability

with XPUPatchForImport(False):
    from test_transformers import (
        TestAttnBias,
        TestSDPA,
        TestSDPAFailureModes,
        TestTransformers,
    )

    @parametrize("nb_heads", [1, 8])
    @parametrize("bias", [True, False])
    def mha_native_args(self, nb_heads, bias):
        B, L, F = 8, 100, 128
        batch_first = True
        fast_path = True
        use_pad_mask = (bias % 2) == 1

        mha = torch.nn.MultiheadAttention(
            embed_dim=F, num_heads=nb_heads, batch_first=batch_first, bias=bias
        ).xpu()
        mha.eval()

        ctx = torch.no_grad if fast_path else contextlib.nullcontext
        with ctx():
            x = torch.randn(B, L, F).xpu()
            if not batch_first:
                x = x.transpose(0, 1)

            pad_mask = None
            if use_pad_mask:
                pad_mask = torch.zeros((B, L), dtype=torch.bool).xpu()

            mha(query=x, key=x, value=x, key_padding_mask=pad_mask)

    def _test_mem_eff_attention_fail_with_batch_size_geq_65536(self):
        query = torch.rand([2**16, 2, 2, 8], device="xpu", dtype=torch.float16)
        key = torch.rand([2**16, 2, 2, 8], device="xpu", dtype=torch.float16)
        value = torch.rand([2**16, 2, 2, 8], device="xpu", dtype=torch.float16)
        with sdpa_kernel(backends=SDPBackend.EFFICIENT_ATTENTION):
            out = F.scaled_dot_product_attention(query, key, value)
        out_cpu = F.scaled_dot_product_attention(query.cpu(), key.cpu(), value.cpu())
        self.assertEqual(out, out_cpu, atol=1e-3, rtol=1e-4)

    def _test_mem_eff_attention_fail_with_batch_size_geq_65536_error(self):
        query = torch.rand([2**16, 2, 2, 8], device="xpu", dtype=torch.float16)
        key = torch.rand([2**16, 2, 2, 8], device="xpu", dtype=torch.float16)
        value = torch.rand([2**16, 2, 2, 8], device="xpu", dtype=torch.float16)
        error_str = (
            r"Efficient attention cannot produce valid seed, "
            r"logsumexp and offset outputs when the batch size exceeds \(65535\)\."
        )
        with self.assertRaisesRegex(RuntimeError, error_str):
            torch._scaled_dot_product_efficient_attention(
                query, key, value, attn_bias=None, compute_log_sumexp=True
            )

    TestSDPAFailureModes.test_mem_eff_attention_fail_with_batch_size_geq_65536 = (
        _test_mem_eff_attention_fail_with_batch_size_geq_65536
    )
    TestSDPAFailureModes.test_mem_eff_attention_fail_with_batch_size_geq_65536_error = (
        _test_mem_eff_attention_fail_with_batch_size_geq_65536_error
    )
    TestTransformers.test_mha_native_args = mha_native_args

instantiate_device_type_tests(
    TestTransformers, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(
    TestSDPAFailureModes, globals(), only_for="xpu", allow_xpu=True
)
instantiate_device_type_tests(TestSDPA, globals(), only_for="xpu", allow_xpu=True)
instantiate_device_type_tests(TestAttnBias, globals(), only_for="xpu", allow_xpu=True)


if __name__ == "__main__":
    run_tests()
