# Owner(s): ["module: intel"]

import contextlib

import torch
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
